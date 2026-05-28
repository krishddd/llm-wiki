# CLAUDE.md — LLM-Wiki v2 Schema

> Read by Claude Code at the start of every session.
> Defines the wiki architecture, memory tiers, conventions, and workflows.
> **Update this file when conventions change.** Companion `AGENTS.md` is the
> agent-facing tool catalogue.

---

## Models

| Role | Model | Strengths |
|------|-------|-----------|
| Summarise, extract | `gemma4:e4b` | Fast, strong instruction-following |
| Reason, route, lint, claims | `qwen3:14b` | Deep reasoning, thinking mode |
| Embeddings | `nomic-embed-text:latest` | 274 MB, MTEB-strong |
| Vision (image captions) | `llava:7b` | Optional — used when ingest_caption_images=true |

All models served via Ollama at `OLLAMA_HOST` (default `http://localhost:11434`).
`MODEL_FAST = MODEL_REASON` is intentional — disables a missing-llama3.2 fallback.

---

## Three-layer architecture

1. **Raw sources** (`wiki/raw/`) — immutable input documents.
2. **The wiki** (`wiki/`) — LLM-generated Markdown across four memory tiers.
3. **The schema** (this file + `AGENTS.md`) — what the LLM is allowed/required to do.

---

## Memory tiers (v2)

| Tier | Where | Lifetime | What goes here |
|---|---|---|---|
| **Working** | in-process state | a single request | retrieved candidates, draft answer |
| **Episodic** | `wiki/episodic/<date>.md` | 14 days (configurable) | every ingest / query / lint event with correlation IDs |
| **Semantic** | `wiki/sources/`, `wiki/entities/` | indefinite, decays | consolidated knowledge, auto-generated entity pages |
| **Procedural** | `wiki/procedures/` + `data/procedures.db` | indefinite | repeated query patterns crystallised into reusable procedures |

**Promotion rules**:
- A query becomes part of episodic on every successful answer.
- A topic that recurs ≥ 3 times across ≥ 14 days of episodic gets auto-promoted to semantic via `promote_episodic_to_semantic` (scheduled daily 04:00 UTC).
- A query pattern that recurs ≥ 5 times with similar retrieval set becomes a procedure via `detect_procedures` (scheduled weekly Sun 06:00 UTC).
- Save-back at query time still produces `wiki/sources/synthesis-<slug>.md` for high-confidence (≥ 0.80) answers with ≥ 2 citations.

---

## Directory layout

```
LLM_Wiki/
├── CLAUDE.md                  ← This file (human-facing)
├── AGENTS.md                  ← Agent-facing tool/resource catalogue
├── src/
│   ├── api.py                 ← FastAPI endpoints
│   ├── ingest.py              ← Ingest pipeline (loaders → summarise → claims → graph)
│   ├── query.py               ← Hybrid retrieval + reflective synthesis + save-back
│   ├── lint.py                ← Health check + auto-fix
│   ├── graph.py               ← KnowledgeGraph (entities, relations, facts) — bi-temporal
│   ├── llm.py                 ← Async Ollama client (cached embeddings)
│   ├── config.py              ← pydantic-settings — all knobs in one place
│   ├── logging_config.py      ← SOC-style JSON logs + audit channel
│   ├── scheduler.py           ← Phase D: APScheduler + JOB_REGISTRY
│   ├── mcp_server.py          ← Phase F3: agent-facing MCP wrapper
│   ├── search/                ← BM25 + dense + RRF + rerank + MMR + multi-query + intent
│   ├── synth/                 ← Answer blocks + per-claim confidence + reflect + followups
│   ├── loaders/               ← Multi-format ingest (PDF/DOCX/HTML/PPTX/XLSX/CSV/MD/TXT)
│   └── wiki/
│       ├── pages.py           ← Page read/write + PageStore
│       ├── synth_page.py      ← Standalone util (used by save-back / promote / crystallize)
│       ├── episodic.py        ← Episodic log + read_episodes + prune
│       ├── promote.py         ← Episodic → semantic auto-promotion
│       ├── procedures.py      ← Procedural memory tier (SQLite + Markdown)
│       ├── reconciler.py      ← Memory-evolution edits to existing pages
│       ├── lifecycle.py       ← Ebbinghaus decay + access reinforcement
│       ├── contradiction_resolver.py ← Composite-score auto-resolution
│       ├── entity_pages.py    ← Auto-generated per-entity pages
│       ├── index_md.py        ← Deterministic regen of wiki/index.md
│       └── log_md.py          ← Append-only operation log
├── wiki/
│   ├── index.md
│   ├── log.md
│   ├── sources/               ← Semantic tier — primary citable content
│   ├── entities/              ← Semantic tier — auto-generated entity pages
│   ├── procedures/            ← Procedural tier — recurring patterns
│   ├── episodic/<date>.md     ← Episodic tier — append-only daily logs
│   ├── archive/               ← Stale low-conf pages moved here by lint auto-fix
│   ├── review/                ← Confidence-gated drafts awaiting human accept
│   │   └── edits/             ← Reconciler-staged edit proposals
│   └── raw/                   ← Immutable source documents
├── data/
│   ├── graph.db               ← SQLite: entities, relations, facts (bi-temporal), page_access
│   ├── procedures.db          ← SQLite: recurring query patterns
│   ├── bm25.pkl
│   └── chroma/                ← ChromaDB persistence (or numpy fallback)
└── logs/
    ├── app.log                ← All events (rotating JSON)
    └── audit.log              ← Filtered audit channel
```

---

## Page conventions (frontmatter)

```yaml
---
title: "Page Title"
kind: source | entity | synthesis | promoted | crystallized | procedure
source: "wiki/raw/file.pdf" | "query-save-back" | "episodic-promotion" | "session-crystallize"
ingested: 2026-05-01
confidence: 0.87
confidence_reason: "..."
tags: [concept, person, org]
entity_refs: ["Entity A", "Entity B"]
context_preamble: "..."     # Anthropic Contextual Retrieval — short doc context
has_tables: true
has_images: false
element_counts: {text: 14, heading: 6, table: 3, image: 0, code: 0}
evolved_by:                  # populated by reconciler when this page is edited by a later source
  - {source: "Foo Doc", action: "refine", date: 2026-05-15}
correlation_ids: [COR-...]   # only on crystallized / promoted
---
```

**Confidence gate**: `>= confidence_threshold` (default 0.60) → `wiki/sources/`, else `wiki/review/`.

**Bi-temporal facts** (separate from page confidence): every claim in the `facts`
table carries `ingested_at`, optional `valid_from`, optional `valid_to`,
`superseded_by`, `last_reinforced`, and `access_count`. **Effective confidence**
is the stored value × Ebbinghaus decay; computed on read.

---

## Workflows

### Ingest

```
load_elements (multi-format)
  → layout_aware_chunks (atomic tables/images)
  → gemma summarise per chunk + extract entities/relations
  → qwen merge (3-tier fallback) + score confidence
  → extraction-signal floor (rich → bumps confidence)
  → contextual preamble (Anthropic) for embedding text
  → write to sources/ or review/
  → upsert entities + relations
  → extract S-P-O claims (qwen) → add_fact()         [v2]
  → contradiction detection vs. related pages
       → on flag: supersede_fact() on older page    [v2]
       → on flag: composite-score auto-resolver     [v2 — Phase E2]
  → reconciler: edit affected pre-existing pages
       → on apply: supersede_fact() if action ∈ {refine, contradict}  [v2]
  → rebuild_index + rebuild_entity_pages
  → episodic_log_entry
```

### Query

```
intent classifier (factual / multi_hop / synthesis / exhaustive)
  → decompose (compound)
  → multi-query paraphrase (RAG-Fusion)
  → HyDE seed for dense
  → hybrid retrieval (BM25 + dense → RRF → FlashRank → graph 2-hop → MMR)
  → mark_accessed() on retrieved pages              [v2 — Phase B3]
  → CRAG relevance filter (drop off-topic)
  → synthesis (numbered citations, [Page]^conf markers, blocks)
  → grounding check + CRAG ceiling
  → reflection critique → optional refinement
  → record_query_pattern() in procedural store      [v2 — Phase C4]
  → save-back if conf ≥ 0.80 ∧ ≥ 2 cits
  → episodic_log_entry
```

### Lint

```
qwen scans first 30 pages
  → JSON report (orphans, stale, missing_entity_pages, contradictions)
  → if auto_fix=True (Phase E1):
       backlink orphans from index.md
       rebuild_entity_pages()
       archive stale low-conf old pages → wiki/archive/
       comment broken cross-references
```

### Scheduled jobs (APScheduler, in-process)

| time UTC | job | toggle |
|---|---|---|
| daily 03:00 | `decay_sweep` | `JOB_DECAY_SWEEP_ENABLED` |
| daily 03:30 | `episodic_prune` | `JOB_EPISODIC_PRUNE_ENABLED` |
| daily 04:00 | `promote_episodic` | `JOB_PROMOTE_EPISODIC_ENABLED` |
| weekly Sun 05:00 | `lint_autofix` | `JOB_LINT_AUTOFIX_ENABLED` |
| weekly Sun 06:00 | `detect_procedures` | `JOB_DETECT_PROCEDURES_ENABLED` |

Manual: `POST /admin/run/{job_name}` runs any registered job once.

---

## Confidence policies

- **Stored confidence** is what the LLM assigned at ingest. Don't mutate it on read.
- **Effective confidence** = stored × `exp(-Δdays / half_life_days)`, floored at 0.05. Half-life default = 90 days.
- **Reinforcement** triggers when a page is accessed ≥ 3 times within a 14-day window. The reinforcement timestamp resets the decay clock.
- **Decay sweep** (scheduled daily) DOES rewrite stored confidence based on `last_reinforced`. Day-to-day reads still compute effective on the fly.

## Supersession lifecycle

- A new source NEVER deletes an old fact. It can only mark it superseded:
  `valid_to = today`, `superseded_by = <new fact id>`.
- Three triggers:
  1. **Reconciler auto-apply**: when `action ∈ {refine, contradict}` lands and `old_text` is matched on the target page.
  2. **Contradiction detector**: when `_detect_contradictions` returns a concrete claim excerpt.
  3. **Auto-resolver** (Phase E2): when a contradiction is detected with composite score margin ≥ 0.2.
- Below the margin → leave both active, surface in `GET /admin/contradictions` for human review.

## Privacy filtering (policy — implementation deferred)

Sources may contain PII / credentials. Apply BEFORE ingest:
- Strip API keys (`sk-...`, `ghp_...`, `xoxb-...`, etc.).
- Strip access tokens, JWTs.
- Strip plaintext passwords.
- Strip private email addresses unless they are public (e.g. paper authors).
- Audit-log every redaction with `PRIVACY_REDACT` event.

This is a documented policy; the redactor module is deferred to a future workstream.

---

## Knowledge graph

- **Entity types**: `PERSON`, `ORG`, `CONCEPT`, `PLACE`, `EVENT`
- **Relation types**: `RELATES_TO`, `PART_OF`, `CONTRADICTS`, `SUPPORTS`, `AUTHORED_BY`, `OCCURRED_IN`
- Fuzzy canonicalization at threshold **95** (raised from 90 for cross-domain safety).
- Reconciler requires **≥ 2 entity overlaps** before considering a page affected (single-entity coincidences ignored).
- 2-hop expansion at retrieval time.

---

## Logging conventions

JSON. Standard fields:
```
timestamp · event_id · correlation_id · severity · component · message · metadata
```

Audit events (always written to `logs/audit.log`):
- `WIKI_WRITE`, `WIKI_WRITE_STAGED`, `WIKI_REVIEW_ACCEPT`, `WIKI_REVIEW_REJECT`
- `CONFIDENCE_LOW`, `CONTRADICTION_DETECTED`
- `STALE_PAGE_DETECTED`, `ORPHAN_PAGE_DETECTED`
- `FACT_SUPERSEDED` (new in v2)

---

## Session checklist (Claude Code at session start)

1. Read this file (`CLAUDE.md`) and `AGENTS.md`.
2. `GET /context/start?days=7` — recent episodic + top pages + open contradictions.
3. Check `wiki/log.md` last 5 entries.
4. Check `wiki/review/` and `wiki/review/edits/` for staged work awaiting human review.
5. `GET /admin/contradictions` for unresolved contradictions.

After significant changes:
- Update this file.
- Run `make test` (if present) and fix failures.
- `POST /lint {auto_fix: true}` to self-heal.

---

*v2 last updated: 2026-05-01 — bi-temporal facts, lifecycle, scheduler, auto-fix, crystallize.*
