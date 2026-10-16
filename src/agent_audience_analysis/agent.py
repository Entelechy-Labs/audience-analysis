"""Audience-analysis agent — Phase 09 strategy system.

Consumes the research phase outputs and produces contract-conforming
``AudienceProfile[]`` artifacts modeling the target audience(s).

Design rules (shared with the Phase 08 research agents):
- The LLM is used only for content judgment; ids, timestamps, dedupe, capping,
  and confidence bounds are deterministic Python.
- Model output is untrusted: it arrives router-validated as JSON, and every
  field is sanitized before entering a contract model.
- Ids are derived deterministically (uuid5) so identical inputs produce
  identical artifacts (idempotent replay, golden tests, checkpoint recovery).
- No network, clock, or randomness: time comes from the injected context
  clock; failures come from the router as duck-typed-normalized errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from linkgen_contracts.common.ids import ProfileId, RunId, TaskId
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import (
    CreatorProfile,
    MarketSignal,
    ResearchRequest,
)
from linkgen_contracts.strategy.models import AudienceProfile
from pydantic import ValidationError

from agent_framework.base import AgentBase
from agent_framework.config import AgentConfig
from agent_framework.context import AgentContext
from agent_framework.errors import (
    AgentError,
    AgentInputError,
    AgentOutputError,
)
from agent_framework.ids import deterministic_id
from agent_framework.sanitize import (
    bounded_str_list,
    clean_text,
    normalize_key,
    safe_confidence,
    stable_probability,
)
from agent_framework.structured import extract_json_dict

from agent_audience_analysis.config import (
    AUDIENCE_OUTPUT_SCHEMA,
    build_audience_analysis_config,
)

# Bounded resource usage (AGENTS rule 11): a 30-day strategy addresses a
# handful of segments, never an unbounded audience taxonomy.
MAX_AUDIENCES = 6
MAX_DEMOGRAPHICS = 10
MAX_DEMOGRAPHIC_KEY = 64
MAX_DEMOGRAPHIC_VALUE = 200

__all__ = [
    "AudienceAnalysisAgent",
    "baseline_audience",
    "build_audiences",
    "build_prompt",
    "dedupe_audiences",
    "normalize_audiences",
    "sanitize_demographics",
    "validate_input",
    "normalize_router_failure",
]


# ---------------------------------------------------------------------------
# Untrusted-input normalization (agent-specific; sanitizers are shared).
# ---------------------------------------------------------------------------

def normalize_audiences(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter the untrusted ``audiences`` array down to dict entries."""
    raw = payload.get("audiences")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentOutputError("model output field 'audiences' must be an array")
    return [entry for entry in raw if isinstance(entry, dict)]


def dedupe_audiences(
    candidates: list[dict[str, Any]], *, cap: int = MAX_AUDIENCES
) -> list[dict[str, Any]]:
    """Dedupe by normalized name; drop unusable entries; cap deterministically.

    A segment needs both a name and a description to be usable downstream, so
    entries missing either are dropped. Duplicates keep the first occurrence
    and the cap keeps the first ``cap`` entries: the result is a
    deterministic function of the input order.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in candidates:
        name = clean_text(entry.get("name"))
        description = clean_text(entry.get("description"))
        key = normalize_key(name)
        if not key or not description:
            continue  # not usable — skip deterministically
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= cap:
            break
    return out


def sanitize_demographics(value: Any) -> dict[str, str]:
    """Sanitize an untrusted demographics mapping into bounded ``str → str``.

    Keys and values are cleaned/truncated; empty keys dropped; entries capped.
    Non-dict input yields an empty mapping (never raises).
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        clean_k = clean_text(key, max_len=MAX_DEMOGRAPHIC_KEY)
        clean_v = clean_text(item, max_len=MAX_DEMOGRAPHIC_VALUE)
        if not clean_k or not clean_v:
            continue
        out[clean_k] = clean_v
        if len(out) >= MAX_DEMOGRAPHICS:
            break
    return out


def _validate_model_list(raw: Any, model: type, label: str) -> list[Any]:
    """Validate an optional list of already-contracted input artifacts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentInputError(f"audience-analysis input '{label}' must be a list")
    validated: list[Any] = []
    for index, item in enumerate(raw):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            raise AgentInputError(
                f"{label}[{index}] failed contract validation at '{location}': "
                f"{first.get('msg', 'invalid')}"
            ) from exc
    return validated


def validate_input(
    input_data: Any,
) -> tuple[ResearchRequest, list[CreatorProfile], list[MarketSignal]]:
    """Validate the agent input against the research-phase contracts.

    Input shape::

        {"request": ResearchRequest,                 # required
         "creators": [CreatorProfile, ...],           # optional
         "signals": [MarketSignal, ...]}              # optional
    """
    if not isinstance(input_data, dict):
        raise AgentInputError(
            "audience-analysis input must be a dict with a 'request' key"
        )
    raw_request = input_data.get("request")
    if raw_request is None:
        raise AgentInputError("audience-analysis input is missing the 'request' object")
    try:
        request = ResearchRequest.model_validate(raw_request)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"research request failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc
    creators = _validate_model_list(input_data.get("creators"), CreatorProfile, "creators")
    signals = _validate_model_list(input_data.get("signals"), MarketSignal, "signals")
    return request, creators, signals


class _RouterFailure(AgentError):
    """Agent-level wrapper for a router failure (category/code preserved)."""


def normalize_router_failure(exc: Exception) -> AgentError:
    """Normalize a router failure without importing provider internals.

    Router errors carry ``category``/``code``/``retryable``; they are preserved
    by duck-typing so ``TaskResult`` mapping keeps retry semantics honest.
    """
    failure = _RouterFailure(f"router call failed: {getattr(exc, 'message', None) or exc}")
    raw_category = getattr(exc, "category", ErrorCategory.INTERNAL)
    failure.category = raw_category if isinstance(raw_category, ErrorCategory) else ErrorCategory.INTERNAL
    failure.code = str(getattr(exc, "code", "router_failure"))
    failure.retryable = bool(getattr(exc, "retryable", False))
    return failure


def build_prompt(
    request: ResearchRequest,
    creators: list[CreatorProfile],
    signals: list[MarketSignal],
) -> str:
    """Deterministic user prompt for the audience-analysis LLM call."""
    lines = [
        "Model the target audience for a LinkedIn content system.",
        f"Focus: {request.focus}",
    ]
    if request.objectives:
        lines.append("Objectives: " + "; ".join(request.objectives))
    if creators:
        lines.append(
            "Creators already researched: "
            + ", ".join(f"{c.name} ({c.handle})" for c in creators[:10])
        )
    if signals:
        lines.append("Market signals observed:")
        lines.extend(f"- {s.description}" for s in signals[:10])
    lines.append(
        "Describe distinct audience segments: who they are, their needs, "
        "interests, and preferred formats. Ground every segment in the "
        "supplied research; rate confidence honestly within [0, 1]; never "
        "invent demographics the research does not imply."
    )
    lines.append("Return JSON with an 'audiences' array.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions of their inputs).
# ---------------------------------------------------------------------------

def build_audiences(
    *,
    run_id: RunId,
    candidates: list[dict[str, Any]],
    captured_at: datetime,
) -> list[AudienceProfile]:
    """Map untrusted, deduped segment entries into contract-valid profiles.

    ``candidates`` must already be deduped (``dedupe_audiences``).
    """
    profiles: list[AudienceProfile] = []
    for entry in candidates:
        name = clean_text(entry.get("name"))
        description = clean_text(entry.get("description"))
        if not name or not description:
            continue
        profiles.append(
            AudienceProfile(
                profile_id=deterministic_id(
                    ProfileId, str(run_id), "audience", normalize_key(name)
                ),
                run_id=run_id,
                name=name,
                description=description,
                demographics=sanitize_demographics(entry.get("demographics")),
                interests=bounded_str_list(entry.get("interests")),
                needs=bounded_str_list(entry.get("needs")),
                preferred_formats=bounded_str_list(entry.get("preferred_formats")),
                confidence=safe_confidence(entry.get("confidence")),
                captured_at=captured_at,
            )
        )
    return profiles


def baseline_audience(
    *, run_id: RunId, request: ResearchRequest, captured_at: datetime
) -> AudienceProfile:
    """The audience derived from the request focus itself.

    Used only when the model produced no usable segment, so the strategy
    phase always receives at least one *input-derived* (never invented)
    audience. The confidence is a stable probability of the run/request, not
    a research claim.
    """
    focus = clean_text(request.focus) or "unspecified"
    return AudienceProfile(
        profile_id=deterministic_id(ProfileId, str(run_id), "audience", normalize_key(focus)),
        run_id=run_id,
        name=f"Primary audience: {focus}",
        description=(
            f"Audience derived from the research focus '{focus}' for run "
            f"{run_id}; refine once segment research is available."
        ),
        demographics={},
        interests=bounded_str_list(request.objectives),
        needs=bounded_str_list([focus]),
        preferred_formats=[],
        confidence=stable_probability(
            "baseline-audience", str(run_id), str(request.request_id)
        ),
        captured_at=captured_at,
    )


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

class AudienceAnalysisAgent(AgentBase):
    """Model the target audience (strategy phase, Phase 09).

    Input (validated against research-phase contracts)::

        {"request": ResearchRequest, "creators"?: [CreatorProfile],
         "signals"?: [MarketSignal]}

    Output::

        {"audiences": [AudienceProfile]}
    """

    def __init__(
        self, context: AgentContext, config: AgentConfig | None = None
    ) -> None:
        super().__init__(
            config if config is not None else build_audience_analysis_config(),
            context,
        )

    def process(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        request, creators, signals = validate_input(input_data)
        payload = self._gather(
            run_id=run_id, task_id=task_id,
            request=request, creators=creators, signals=signals,
        )

        audiences = build_audiences(
            run_id=run_id,
            candidates=dedupe_audiences(normalize_audiences(payload)),
            captured_at=self.context.now(),
        )
        if not audiences:
            audiences = [
                baseline_audience(
                    run_id=run_id, request=request, captured_at=self.context.now()
                )
            ]

        return {
            "audiences": [profile.model_dump(mode="json") for profile in audiences]
        }

    def _gather(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        request: ResearchRequest,
        creators: list[CreatorProfile],
        signals: list[MarketSignal],
    ) -> dict[str, Any]:
        """Call the router for structured audience segments (single LLM hop).

        Fault markers in the prompt (mock mode) or a failing adapter surface
        as normalized agent errors; they never escape as raw provider errors.
        """
        prompt = build_prompt(request, creators, signals)
        llm_request = self.build_llm_request(
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            json_schema=AUDIENCE_OUTPUT_SCHEMA,
        )
        try:
            result = self.context.router.complete_json(llm_request, AUDIENCE_OUTPUT_SCHEMA)
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise normalize_router_failure(exc) from exc
        return extract_json_dict(result)
