"""Typed definition of the audience-analysis agent.

The ``AgentConfig`` is the definition-time contract of this agent: mission,
model role, allowed/forbidden tools, output contracts, and the JSON schema
its LLM calls must satisfy. The router validates model output against this
schema (fail closed) before the agent ever sees it.
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import AgentConfig
from linkgen_contracts.llm.models import LLMRole, ResponseFormat

AUDIENCE_ANALYSIS_SLUG = "audience-analysis"

# JSON schema for the model's structured output. Restricted to the router's
# documented schema subset (type/properties/items/required/enum/const) so it
# fails closed rather than being silently ignored. Ids and timestamps are
# NEVER requested from the model — deterministic Python derives them.
AUDIENCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "audiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "demographics": {"type": "object"},
                    "interests": {"type": "array", "items": {"type": "string"}},
                    "needs": {"type": "array", "items": {"type": "string"}},
                    "preferred_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """You are the audience-analysis agent of the LinkGen content system.
Mission: model the target audience(s) for the requested focus from the
completed research.
Describe distinct audience segments: who they are, what they struggle with,
what they need, what they care about, and which formats they prefer. Ground
every segment in the supplied research signals and creators; rate confidence
honestly and never above what the research supports. Do not invent
demographics the research does not imply.
Respond only with JSON matching the requested schema."""


def build_audience_analysis_config() -> AgentConfig:
    """Build the definition-time configuration of the audience agent."""
    return AgentConfig(
        slug=AUDIENCE_ANALYSIS_SLUG,
        model_role=LLMRole.REASONING,
        output_contracts=[
            "AudienceProfile",
        ],
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        max_steps=8,
        tools=["web-search", "browser"],
        forbidden_tools=["publish", "schedule", "linkedin-post"],
        response_format=ResponseFormat.JSON,
        json_schema=AUDIENCE_OUTPUT_SCHEMA,
        description="Model the target audience (strategy phase).",
        version="1.0.0",
    )
