"""Unit tests for audience normalization (agent-specific logic).

Generic sanitizer behavior is covered by agent-framework's tests; here we
test how this agent applies it to model-proposed audience segments.
"""

from __future__ import annotations

import pytest

from agent_audience_analysis.agent import (
    MAX_AUDIENCES,
    dedupe_audiences,
    normalize_audiences,
)
from agent_framework.errors import AgentOutputError


def test_normalize_audiences_filters_and_validates() -> None:
    payload = {"audiences": [{"name": "a"}, "junk", 5, {"name": "b"}]}
    assert normalize_audiences(payload) == [{"name": "a"}, {"name": "b"}]
    assert normalize_audiences({}) == []
    assert normalize_audiences({"audiences": None}) == []
    with pytest.raises(AgentOutputError):
        normalize_audiences({"audiences": "nope"})


def test_dedupe_audiences_by_normalized_name() -> None:
    candidates = [
        {"name": " Founders ", "description": "d1"},
        {"name": "founders", "description": "duplicate"},
        {"name": "Operators", "description": "d2"},
        {"name": "", "description": "dropped — no name"},
        {"name": "   ", "description": "dropped — blank name"},
        {"name": "missing description", "description": "  "},
    ]
    out = dedupe_audiences(candidates)
    assert [entry["name"] for entry in out] == [" Founders ", "Operators"]


def test_dedupe_audiences_caps_deterministically() -> None:
    candidates = [
        {"name": f"Segment {i}", "description": f"description {i}"}
        for i in range(MAX_AUDIENCES + 3)
    ]
    out = dedupe_audiences(candidates)
    assert len(out) == MAX_AUDIENCES
    assert out[0]["name"] == "Segment 0"
    assert out[-1]["name"] == f"Segment {MAX_AUDIENCES - 1}"
