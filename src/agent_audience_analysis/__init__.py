"""audience-analysis — models the target audience (Phase 09).

Strategy-phase agent of the LinkGen content-intelligence system. Consumes the
research phase outputs (``ResearchRequest`` plus optional ``CreatorProfile[]``
and ``MarketSignal[]``) and produces contract-conforming ``AudienceProfile[]``
artifacts through the shared agent framework and ModelRouter (mock mode by
default).
"""

from __future__ import annotations

from agent_audience_analysis.config import (
    AUDIENCE_ANALYSIS_SLUG,
    AUDIENCE_OUTPUT_SCHEMA,
    build_audience_analysis_config,
)
from agent_audience_analysis.agent import AudienceAnalysisAgent
from agent_audience_analysis.version import __version__

__all__ = [
    "__version__",
    "AUDIENCE_ANALYSIS_SLUG",
    "AUDIENCE_OUTPUT_SCHEMA",
    "AudienceAnalysisAgent",
    "build_audience_analysis_config",
]
