"""Agent-level tests: normal cases, edge cases, and malformed inputs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_audience_analysis.agent import (
    AudienceAnalysisAgent,
    baseline_audience,
    build_audiences,
    build_prompt,
    sanitize_demographics,
    validate_input,
)
from agent_audience_analysis.config import AUDIENCE_ANALYSIS_SLUG
from agent_framework.errors import AgentInputError
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import CreatorProfile, MarketSignal, ResearchRequest
from linkgen_contracts.strategy.models import AudienceProfile
from linkgen_contracts.workflow.models import TaskStatus

from conftest import (
    TASK_ID,
    make_creator,
    make_input,
    make_request,
    make_signal,
)

FIXED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)

CANDIDATES = [
    {
        "name": "Ops leads at mid-market SaaS",
        "description": "Own onboarding rollout; measured on time-to-value.",
        "demographics": {"role": "Operations lead", "seniority": "mid"},
        "interests": ["process design", "activation"],
        "needs": ["checklists that fit their product"],
        "preferred_formats": ["carousel", "checklist posts"],
        "confidence": 0.72,
    },
    {
        "name": "ops leads at mid-market saas",  # duplicate name
        "description": "Duplicate — dropped by dedupe.",
    },
    {"name": "", "description": "dropped — no name"},
    {"name": "nameless segment", "description": "   "},
]


class TestInputValidation:
    def test_valid_input_round_trips(self) -> None:
        request, creators, signals = validate_input(make_input())
        assert isinstance(request, ResearchRequest)
        assert len(creators) == 1 and isinstance(creators[0], CreatorProfile)
        assert len(signals) == 1 and isinstance(signals[0], MarketSignal)

    def test_minimal_input_accepted(self) -> None:
        request, creators, signals = validate_input({"request": make_request()})
        assert creators == [] and signals == []

    def test_non_dict_input_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(["nope"])

    def test_missing_request_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input({"signals": []})

    def test_empty_focus_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(request=make_request(focus="   ")))

    def test_invalid_creator_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(creators=[{"handle": ""}]))


class TestNormalRun:
    def test_process_produces_contract_valid_output(
        self, agent: AudienceAnalysisAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        assert set(output) == {"audiences"}
        profiles = [AudienceProfile.model_validate(item) for item in output["audiences"]]
        assert profiles  # never an empty result (baseline fallback)
        for profile in profiles:
            assert str(profile.profile_id).startswith("profile_")
            assert profile.run_id == run_id
            assert profile.name and profile.description
            assert 0.0 <= profile.confidence <= 1.0
            assert isinstance(profile.demographics, dict)
            for key, value in profile.demographics.items():
                assert isinstance(key, str) and isinstance(value, str)

    def test_run_reports_success(
        self, agent: AudienceAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data=make_input())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.errors == []

    def test_config_declares_expected_boundaries(
        self, agent: AudienceAnalysisAgent
    ) -> None:
        assert agent.config.slug == AUDIENCE_ANALYSIS_SLUG
        assert "publish" in agent.config.forbidden_tools
        assert "schedule" in agent.config.forbidden_tools
        assert agent.config.response_format.value == "json"
        assert "AudienceProfile" in agent.config.output_contracts


class TestRunFailures:
    def test_malformed_input_maps_to_failed_result(
        self, agent: AudienceAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=TASK_ID, input_data="junk")
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_missing_request_maps_to_failed_result(
        self, agent: AudienceAnalysisAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data={})
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "agent_input_invalid"


class TestArtifactBuilders:
    def test_dedupe_keeps_first_occurrence_and_drops_unusable(self) -> None:
        from agent_audience_analysis.agent import dedupe_audiences

        kept = dedupe_audiences(CANDIDATES)
        assert len(kept) == 1
        assert kept[0]["name"] == "Ops leads at mid-market SaaS"

    def test_build_audiences_maps_and_sanitizes(self, run_id) -> None:
        profiles = build_audiences(
            run_id=run_id, candidates=CANDIDATES[:1], captured_at=FIXED_AT
        )
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.confidence == 0.72
        assert profile.demographics == {
            "role": "Operations lead",
            "seniority": "mid",
        }
        assert profile.interests == ["process design", "activation"]
        assert profile.needs == ["checklists that fit their product"]
        assert profile.preferred_formats == ["carousel", "checklist posts"]
        assert profile.captured_at == FIXED_AT
        # Ids derive from (run, normalized name) — stable, never random.
        again = build_audiences(
            run_id=run_id, candidates=CANDIDATES[:1], captured_at=FIXED_AT
        )
        assert again[0].profile_id == profile.profile_id

    def test_sanitize_demographics_bounds_and_drops(self) -> None:
        assert sanitize_demographics("not a dict") == {}
        assert sanitize_demographics({"": "value"}) == {}
        assert sanitize_demographics({"role": ""}) == {}
        assert sanitize_demographics({"role": 42}) == {}  # non-string dropped
        many = {f"key{i}": f"value{i}" for i in range(20)}
        assert len(sanitize_demographics(many)) == 10

    def test_out_of_range_confidence_is_clamped(self, run_id) -> None:
        profiles = build_audiences(
            run_id=run_id,
            candidates=[
                {"name": "Wide", "description": "d", "confidence": 9.5}
            ],
            captured_at=FIXED_AT,
        )
        assert profiles[0].confidence == 1.0

    def test_baseline_audience_derives_from_focus(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        profile = baseline_audience(
            run_id=run_id, request=request, captured_at=FIXED_AT
        )
        assert request.focus in profile.name
        assert profile.description
        assert 0.0 <= profile.confidence <= 1.0
        assert profile.profile_id == baseline_audience(
            run_id=run_id, request=request, captured_at=FIXED_AT
        ).profile_id

    def test_prompt_contains_focus_and_ground_rules(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        prompt = build_prompt(request, [], [])
        assert request.focus in prompt
        assert "never invent demographics" in prompt
        assert "Return JSON with an 'audiences' array." in prompt
