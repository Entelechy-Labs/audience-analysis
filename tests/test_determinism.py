"""Determinism and id-stability regression tests (golden path).

The same input, clock, and router must produce identical artifacts; different
run ids must produce different derived ids (no cross-run bleed).
"""

from __future__ import annotations

from agent_audience_analysis.agent import AudienceAnalysisAgent
from agent_framework.ids import deterministic_id
from linkgen_contracts.common.ids import ProfileId, RunId, TaskId

from conftest import RUN_ID, TASK_ID, make_context, make_input


def _run_agent(payload: dict, *, run_id: str = RUN_ID) -> dict:
    agent = AudienceAnalysisAgent(context=make_context())
    return agent.process(
        run_id=RunId(run_id), task_id=TaskId(TASK_ID), input_data=payload
    )


def test_same_input_same_output() -> None:
    first = _run_agent(make_input())
    second = _run_agent(make_input())
    assert first == second


def test_profile_ids_are_derived_deterministically() -> None:
    output = _run_agent(make_input())
    for profile in output["audiences"]:
        expected = deterministic_id(
            ProfileId, RUN_ID, "audience", profile["name"].lower()
        )
        assert profile["profile_id"] == expected


def test_different_run_id_yields_different_ids() -> None:
    other_run = "run_" + "e4" * 16
    base = _run_agent(make_input())
    other = _run_agent(make_input(run_id=other_run), run_id=other_run)
    base_ids = {p["profile_id"] for p in base["audiences"]}
    other_ids = {p["profile_id"] for p in other["audiences"]}
    if base_ids and other_ids:
        assert base_ids != other_ids


def test_same_run_same_artifacts_across_contexts() -> None:
    # Fresh agent + fresh context, same fixed clock/router: byte-identical.
    agent_one = AudienceAnalysisAgent(context=make_context())
    agent_two = AudienceAnalysisAgent(context=make_context())
    kwargs = dict(run_id=RunId(RUN_ID), task_id=TaskId(TASK_ID))
    assert agent_one.process(input_data=make_input(), **kwargs) == (
        agent_two.process(input_data=make_input(), **kwargs)
    )
