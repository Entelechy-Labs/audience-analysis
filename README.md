# audience-analysis

Strategy-phase agent of the **LinkGen** content-intelligence and publishing
system.

## Purpose

The specialist agent responsible for modeling the target audience: it
consumes the research phase outputs (`ResearchRequest` plus optional
`CreatorProfile[]` / `MarketSignal[]`) and produces contract-conforming
`AudienceProfile[]` segments — who they are, their needs, interests, and
preferred formats.

Design highlights (see `AGENTS.md` in the parent for the master rules):

- **Deterministic by construction** — ids are derived with uuid5
  (`agent_framework.ids.deterministic_id`), time comes from the injected
  context clock, and outputs are pure functions of the input.
- **LLM output is untrusted** — model JSON arrives router-validated and is
  sanitized field-by-field (including the free-form `demographics` mapping,
  which is bounded to clean string keys/values); segments without a
  name/description are dropped, duplicates deduped by normalized name, and
  the segment list is capped.
- **Focus-derived baseline** — when the model yields no usable segment, the
  agent emits one profile derived from the request focus itself, so the
  strategy phase never receives an empty audience list.
- **Mock-first ($0 development)** — the default provider is the offline
  deterministic `MockLLMProvider` via the `ModelRouter`.
- **No privileged tools** — no publishing/scheduling authority.

## Input / output contract

Input (validated against research-phase contracts):

```json
{"request": {"request_id": "...", "run_id": "...", "focus": "..."},
 "creators": [{...CreatorProfile...}],
 "signals": [{...MarketSignal...}]}
```

`creators` and `signals` are optional; `request` is required.

Output:

```json
{"audiences": [{...AudienceProfile...}]}
```

## Boundaries

- Output contracts: `AudienceProfile`.
- Forbidden tools: `publish`, `schedule`, `linkedin-post`.
- Depends only on `linkgen_contracts`, `linkgen_config`, `agent-framework`,
  and the `ModelRouter` interface from `providers/llm`.

## Testing

Run local tests with:

    python -m pytest

## Status

Phase 09 (strategy system) — implemented at 1.0.0.

## Publishing authority

No publishing/scheduling authority; publishing privilege belongs to the
publishing subsystem (publisher) only.
