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
| **Quantitative reasoning specialist** | `vibethinker:3b` | AIME-class maths / STEM / code; routed to ADAPTIVELY for quantitative questions |
| Embeddings | `nomic-embed-text:latest` | 274 MB, MTEB-strong |
| STEM embeddings (optional) | `bge-m3` | Domain-routed embedder for maths/science/econ/eng; enable with `EMBED_STEM_ENABLED` |
| Vision (image captions) | `llava:7b` | Optional — used when ingest_caption_images=true |

All models served via Ollama at `OLLAMA_HOST` (default `http://localhost:11434`).
`MODEL_FAST = MODEL_REASON` is intentional — disables a missing-llama3.2 fallback.

### Multi-provider LLM fleet (v4, `src/providers.py`)

Each role can be routed to a hosted, OpenAI-compatible provider instead of
Ollama via `PROVIDER_<ROLE>` (`summary` / `reason` / `fast` / `solver` / `embed` /
`vision`), default `ollama`. Vision routes the llava image-caption role to a
multimodal provider via the OpenAI `image_url` content schema. Providers: `groq` (LPU-fast open weights), `github` (GitHub Models,
gpt-4.1 family), `gemini` (Google AI Studio; the only provider wired for embeddings).
`resolve_chat_provider()` returns `None` (→ Ollama) when the role is `ollama` or its
key/model is unset, so routing is opt-in and self-healing. A provider HTTP error is
re-raised as `OllamaError` so the existing role fallbacks (e.g. qwen→llama) still fire.
Keys (`GROQ_API_KEY`, `GITHUB_MODELS_TOKEN`, `GOOGLE_GENAI_API_KEY`) are read from env
ONLY — never committed.

### Adaptive model routing (VibeThinker)

`model_solver` (default `vibethinker:3b`, [WeiboAI/VibeThinker](https://github.com/WeiboAI/VibeThinker))
is a tiny reasoning specialist: world-class on competition maths / STEM / code, but
**weak on broad knowledge** (the authors say so). So it is NOT a general synthesizer
replacement — it is routed to ONLY for quantitative questions via a **reason→format**
two-stage:

1. `src/search/domain.py` detects the cognition required (general / math / science /
   economics / engineering) by heuristic regex over the question + retrieved context.
2. If quantitative (`needs_solver()` True) and `route_solver_enabled`, VibeThinker
   does the step-by-step derivation (`OllamaClient.solver()`, temp 0.6, top_p 0.95).
   Its `<think>…</think>` trace is stripped (`strip_think()`).
3. `qwen3:14b` then formats + cites that verified reasoning into the standard
   JSON/citation schema — so grounding, per-claim confidence and save-back are unchanged.

Any solver failure (model not installed, timeout) silently falls back to qwen-only
synthesis. `QueryResult.reasoner` records `"qwen"` or `"solver"`.

**Serving:** VibeThinker ships for vLLM / SGLang / transformers. To use it in this
Ollama stack, either pull a GGUF quant (`ollama create vibethinker:3b -f Modelfile`
with `temperature 0.6`, `top_p 0.95`, `num_ctx 40960`) or run a vLLM sidecar. Set
`model_solver=""` to disable routing entirely.

### Domain-specialized STEM embeddings (optional)

`model_embed_stem` (default `bge-m3`, off unless `EMBED_STEM_ENABLED=true`) is a
stronger embedder for notation-heavy content. When enabled, `DomainRoutedDenseIndex`
(`src/search/dense_router.py`) keeps a **separate** STEM dense collection
(`chroma_stem`) — two embedders mean two incompatible vector spaces, so they cannot
share one collection. Routing: quantitative pages (domain ∈ {math, science, economics,
engineering}) are indexed into BOTH general and STEM collections; quantitative queries
are served by the STEM collection (each index embeds the query with its own model, so
spaces stay consistent). `hybrid_search` calls `route_search()` when present; a plain
`DenseIndex` is unchanged. Disabled = transparent passthrough to the general index.
Enabling requires `ollama pull bge-m3` and re-ingesting (or rebuilding) to populate
the STEM collection.

### Best-of-best RAG package (v5)

Five techniques layered onto the existing pipeline (each flag-gated, on by default):

| Technique | Where | Flag |
|---|---|---|
| **Small-to-big retrieval** — rerank/synthesise the matched 1500-char sub-chunks (±neighbours) instead of `page[:4000]`; `src/search/chunks.py` re-derives the exact index-time chunks | `hybrid.py` | `QUERY_CHUNK_CONTEXT` |
| **Doc2Query** (Nogueira & Lin) — index the questions each doc answers as `<pid>#hq` so question-phrased queries match declarative text | `ingest.py` | `INGEST_DOC2QUERY` |
| **Lost-in-the-middle reorder** (Liu et al. 2023) — ends-load synthesis context: best page first, runner-up last | `query.py` | `QUERY_LITM_REORDER` |
| **NLI-lite claim verification** — one batched gemma call judges each cited claim against its cited snippet; unsupported ×0.35 confidence | `synth/verify.py` | `QUERY_CLAIM_VERIFY` |
| **Machine-page down-weight** — synthesis/promoted/crystallized pages score ×0.85 at rerank so save-backs never outrank primary sources (anti-feedback-loop) | `hybrid.py` | `RETRIEVAL_SYNTH_DOWNWEIGHT` |
| **RAPTOR-lite topics** (Sarthi et al. 2024 / GraphRAG communities) — weekly greedy-cosine clustering of live pages → `topic-*.md` overview pages (kind `topic`) answering corpus-level questions | `wiki/topics.py` | `JOB_BUILD_TOPICS_ENABLED`, `TOPICS_MIN_CLUSTER`, `TOPICS_MAX`, `TOPICS_SIM_THRESHOLD` |

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

The wiki conforms to **Google's Open Knowledge Format (OKF) v0.1**
([spec](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)):
every concept page carries `type` (OKF's one required field), plus the recommended
`description`, `resource`, `tags`, and `timestamp`. `write_page()` stamps `type`
(mapped from `kind` via `OKF_TYPE_BY_KIND`), `description` (first prose sentence of
the body when absent) and `timestamp` (ISO 8601 last-modification) centrally, so every
writer conforms automatically. `index.md` / `log.md` are OKF reserved filenames —
never concept pages (`PageStore.iter_pages` skips them). The root `index.md` declares
`okf_version: "0.1"` and lists pages as `- [Title](path) - description`. `log.md` uses
date-grouped, newest-first entries with bold action keywords. Body cross-links are
bundle-relative markdown links (`[Title](/sources/foo.md)`) — NOT Obsidian wikilinks —
so OKF consumers can treat them as graph edges. `scripts/migrate_okf.py` re-stamps
pre-OKF pages (idempotent).

```yaml
---
title: "Page Title"
kind: source | entity | synthesis | promoted | crystallized | procedure
type: "Source Document" | "Person|Organization|Concept|Place|Event" | "Synthesis" | ...   # OKF required
description: "One-sentence summary (first prose sentence of body if not supplied)"       # OKF recommended
resource: "wiki/raw/file.pdf"        # OKF URI of the underlying asset (mirrors source)
timestamp: "2026-07-15T16:51:36+00:00"  # OKF last-modification; auto-stamped on write
source: "wiki/raw/file.pdf" | "query-save-back" | "episodic-promotion" | "session-crystallize"
ingested: 2026-05-01
confidence: 0.87
confidence_reason: "..."
domain: general | math | science | economics | engineering   # stamped at ingest; drives adaptive routing
chunk_strategy: dense | narrative | balanced | fixed          # agentic-ingestion chunk plan used
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
  → privacy redaction (strip API keys / JWTs / private keys / passwords)  [PRIVACY_REDACT]
  → agentic plan: adaptive chunk size/overlap from structure + density   [agentic ingestion]
  → layout_aware_chunks (atomic tables/images, plan-driven target/overlap)
  → gemma summarise per chunk + extract entities/relations
  → qwen merge (3-tier fallback) + score confidence
  → extraction-signal floor (rich → bumps confidence)
  → contextual preamble (Anthropic) for embedding text
  → Doc2Query: gemma generates the questions the doc answers → indexed as <pid>#hq   [v5]
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

`POST /query` now enters through the **agentic orchestrator** (`agentic_rag/orchestrator.py`):
it assesses question complexity and sends factual/simple queries to the fast single-pass
path (zero overhead) while multi_hop/synthesis/exhaustive go through the iterative agentic
loop (plan → fanout → sufficient-context check → gap rewriter → repeat). Request flag
`agentic`: omit = auto-decide (default), `true` = force loop, `false` = force single-pass.
Disable globally with `AGENTIC_ENABLED=false`. Both paths share the synthesis below.

```
intent classifier (factual / multi_hop / synthesis / exhaustive)
  → decompose (compound)
  → multi-query paraphrase (RAG-Fusion)
  → HyDE seed for dense
  → hybrid retrieval (BM25 + dense → RRF → FlashRank → graph 2-hop → MMR)
       → small-to-big: rerank/synthesise the MATCHED sub-chunks (±neighbours),  [v5]
         not page[:4000] — src/search/chunks.py re-derives index-time chunks
       → machine-page down-weight (synthesis/promoted/crystallized ×0.85)      [v5]
  → mark_accessed() on retrieved pages              [v2 — Phase B3]
  → CRAG relevance filter (drop off-topic)
  → adaptive model routing: if quantitative (maths/econ/science/eng),     [VibeThinker]
       VibeThinker reasons step-by-step → qwen formats + cites the result
  → multimodal expansion: surface media nodes linked to retrieved        [GRAPH_MULTIMODAL_NODES]
       entities (tables/figures/code) into context + related_media
  → lost-in-the-middle reorder: ends-load context (best first, runner-up last)  [v5]
  → synthesis (numbered citations, [Page]^conf markers, blocks)
  → grounding check + CRAG ceiling
  → NLI-lite claim verification: ONE batched gemma call judges each cited claim  [v5]
       against its cited snippet (supported/partial/unsupported) → recalibrates
       per-claim + overall confidence (catches "right page, wrong claim")
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
| weekly Sun 07:30 | `build_topics` (RAPTOR-lite topic overviews) | `JOB_BUILD_TOPICS_ENABLED` |

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

## Agentic ingestion (implemented — `src/agentic_ingest.py`)

`plan_ingest()` inspects each document's structure (element kinds/counts, structural
density, text length) and a content sample, then picks an adaptive chunk plan instead
of the fixed 6000-char target:
- **dense** (STEM domain, formulas, or ≥4 tables/code blocks) → ~3000 chars + ~10%
  overlap, so notation/tables stay with their explanation.
- **narrative** (long prose, low density) → ~7500 chars + ~2% overlap.
- **balanced** → existing defaults.

Heuristic-first (on by default, zero LLM cost). Optional gemma refinement via
`INGEST_PLANNING_LLM` (one extra call per doc). All sizes clamped to [1500, 9000] /
[80, 600]. The chosen strategy is recorded in frontmatter as `chunk_strategy`.

## Privacy filtering (implemented — `src/privacy.py`)

Implemented in `src/privacy.py`. `redact_text()` is applied to raw element text in
`ingest_file()` BEFORE it reaches the summariser / claims / graph / embeddings /
on-disk page. Each secret becomes a typed `[REDACTED:<cat>]` placeholder so prose
stays coherent.
- Strips API keys (`sk-...`, `ghp_...`, `xox[baprs]-...`, `AKIA...`, `AIza...`, GitLab PATs).
- Strips JWTs (`eyJ…`) and PEM private-key blocks.
- Strips plaintext passwords in `password: …` / `pwd=…` form (field name preserved).
- Emails are PII but public author emails are legitimate content → opt-in via
  `INGEST_REDACT_EMAILS` (default off).
- Audit-logs every redaction with a `PRIVACY_REDACT` event carrying per-category counts.

Toggles: `INGEST_REDACT_SECRETS` (default on), `INGEST_REDACT_EMAILS` (default off).
Conservative by design — high-precision patterns only, to avoid corrupting prose.

---

## Knowledge graph

- **Entity types**: `PERSON`, `ORG`, `CONCEPT`, `PLACE`, `EVENT`
- **Relation types**: `RELATES_TO`, `PART_OF`, `CONTRADICTS`, `SUPPORTS`, `AUTHORED_BY`, `OCCURRED_IN`
- **Media nodes** (multimodal graph Phase 1, `GRAPH_MULTIMODAL_NODES`, default off):
  `media_nodes` (kinds `table|image|code|formula`) + `media_entities` edges
  (`DEPICTS|MEASURES|DEFINES|REFERENCES`). Populated at ingest, each embedded as its
  own dense unit (`<pid>#media#<n>`), linked to entities by name presence. Data-only
  in Phase 1 — retrieval through media nodes is a later phase. See
  `docs/design/multimodal-graph.md`.
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
