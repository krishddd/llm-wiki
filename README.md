# llm-wiki

> A self-healing local knowledge base where a local LLM compounds your
> documents across four memory tiers — working, episodic, semantic, and
> procedural — with bi-temporal facts, automatic contradiction resolution,
> and scheduled memory maintenance.

`llm-wiki` is a FastAPI service that turns a folder of raw documents into a
**continuously self-organising Markdown wiki**. Drop PDFs, DOCX, PPTX, XLSX,
HTML, or Markdown into the ingest endpoint and the system extracts entities,
claims, and relations; writes confidence-scored pages; and keeps them honest
over time through bi-temporal fact tracking, Ebbinghaus decay, and weekly
self-lint runs.

The schema is fully described in [`CLAUDE.md`](./CLAUDE.md) and the agent
tool catalogue in [`AGENTS.md`](./AGENTS.md).

---

## Why this exists

Most "chat-with-your-docs" stacks throw documents into a vector store and
walk away. After three months they are full of stale claims, duplicate
entities, and dangling references. `llm-wiki` treats the knowledge base as a
**living artefact that has to be maintained** — it promotes recurring ideas,
decays unreinforced ones, supersedes facts when newer sources contradict
older ones, and crystallises repeated query patterns into reusable
procedures.

---

## The four memory tiers

| Tier           | Where                              | Lifetime          | Contents                                              |
|----------------|------------------------------------|-------------------|-------------------------------------------------------|
| **Working**    | in-process state                   | one request       | retrieved candidates, draft answer                    |
| **Episodic**   | `wiki/episodic/<date>.md`          | 14 days (config)  | every ingest / query / lint event with correlation IDs|
| **Semantic**   | `wiki/sources/`, `wiki/entities/`  | indefinite, decays| consolidated pages, auto-generated entity pages       |
| **Procedural** | `wiki/procedures/` + `procedures.db`| indefinite        | recurring query patterns crystallised into procedures |

**Promotion rules:**
- A query becomes part of episodic on every successful answer.
- A topic that recurs ≥ 3 times across ≥ 14 days of episodic auto-promotes
  to semantic via the daily `promote_episodic_to_semantic` job (04:00 UTC).
- A query pattern recurring ≥ 5 times with a similar retrieval set becomes a
  procedure via the weekly `detect_procedures` job (Sun 06:00 UTC).
- A high-confidence answer (≥ 0.80, ≥ 2 citations) is saved back as
  `wiki/sources/synthesis-<slug>.md` immediately at query time.

---

## The ingest pipeline

```
file (PDF/DOCX/PPTX/XLSX/HTML/MD/TXT)
   │
   ▼
load_elements                     ← multi-format loaders, structure-aware
   │
   ▼
layout_aware_chunks               ← atomic tables / images, semantic blocks
   │
   ▼
gemma summarise per chunk         ← fast LLM extracts entities + relations
   │
   ▼
qwen merge (3-tier fallback)      ← reasoning LLM consolidates + scores conf
   │
   ▼
extraction-signal floor           ← rich text → confidence bump
   │
   ▼
contextual preamble (Anthropic)   ← short doc context attached to chunks
   │
   ▼
confidence gate
   │
   ├─ ≥ 0.60 → wiki/sources/<slug>.md
   └─ <  0.60 → wiki/review/<slug>.md   (awaits human accept)
   │
   ▼
upsert entities + relations into graph.db
   │
   ▼
extract S-P-O claims (qwen)  →  add_fact(valid_from=today)
   │
   ▼
contradiction detection vs related pages
   │
   ├─ concrete contradiction → supersede_fact() on older page
   └─ composite-score auto-resolver (margin ≥ 0.2) → keep winner, mark loser
   │
   ▼
reconciler — edits pre-existing pages that overlap on ≥ 2 entities
   │  refines, contradicts, or adds context; staged in wiki/review/edits/
   │
   ▼
rebuild_index + rebuild_entity_pages
   │
   ▼
episodic_log_entry (correlation_id)
```

Every step is logged in JSON to `logs/app.log`; security-relevant events
(writes, accepts/rejects, contradictions, supersessions) also go to
`logs/audit.log`.

---

## The query pipeline

```
user question
   │
   ▼
intent classifier                 ← factual / multi_hop / synthesis / exhaustive
   │
   ▼
decompose (if compound)
   │
   ▼
multi-query paraphrase            ← RAG-Fusion: N rewrites
HyDE seed for dense retrieval     ← LLM hallucinates a hypothetical doc
   │
   ▼
hybrid retrieval
   ├─ BM25 over wiki/sources/
   ├─ dense over Chroma (or numpy fallback)
   ├─ RRF fuse
   ├─ FlashRank cross-encoder rerank
   ├─ graph 2-hop expansion via entity links
   └─ MMR diversification
   │
   ▼
mark_accessed() on every retrieved page → reinforces lifecycle counter
   │
   ▼
CRAG relevance filter             ← drop off-topic candidates
   │
   ▼
synthesis
   ├─ numbered citations
   ├─ [Page]^conf markers per claim
   └─ structured answer blocks
   │
   ▼
grounding check + CRAG ceiling    ← detect ungrounded statements
   │
   ▼
reflection critique → optional refinement
   │
   ▼
record_query_pattern() in procedural store
   │
   ▼
save-back if confidence ≥ 0.80 ∧ citations ≥ 2
   │
   ▼
episodic_log_entry
```

---

## Confidence and decay

- **Stored confidence** is what the LLM assigned at ingest time; reads do
  not mutate it.
- **Effective confidence** = stored × `exp(-Δdays / half_life_days)`, floored
  at 0.05. Default half-life is 90 days.
- **Reinforcement** triggers when a page is accessed ≥ 3 times within a
  14-day window; the reinforcement timestamp resets the decay clock.
- The **decay sweep** (daily 03:00 UTC) rewrites stored confidence based on
  `last_reinforced`.

Bi-temporal facts carry `ingested_at`, optional `valid_from`, optional
`valid_to`, `superseded_by`, `last_reinforced`, and `access_count`. A new
source can never *delete* an old fact — only mark it superseded by setting
`valid_to = today` and `superseded_by = <new_fact_id>`.

Three triggers can supersede:
1. **Reconciler auto-apply** when `action ∈ {refine, contradict}` lands and
   the old text matches.
2. **Contradiction detector** when `_detect_contradictions` returns a
   concrete claim excerpt.
3. **Auto-resolver** (Phase E2) when the composite-score margin ≥ 0.2;
   sub-margin cases stay surfaced in `GET /admin/contradictions` for human
   review.

---

## Scheduled jobs

In-process APScheduler runs the following by default (each toggleable via env
var). Manual one-off runs available via `POST /admin/run/{job_name}`.

| UTC time         | Job                  | Toggle env var                     |
|------------------|----------------------|------------------------------------|
| daily 03:00      | `decay_sweep`        | `JOB_DECAY_SWEEP_ENABLED`          |
| daily 03:30      | `episodic_prune`     | `JOB_EPISODIC_PRUNE_ENABLED`       |
| daily 04:00      | `promote_episodic`   | `JOB_PROMOTE_EPISODIC_ENABLED`     |
| weekly Sun 05:00 | `lint_autofix`       | `JOB_LINT_AUTOFIX_ENABLED`         |
| weekly Sun 06:00 | `detect_procedures`  | `JOB_DETECT_PROCEDURES_ENABLED`    |

---

## Models

| Role                         | Model                       | Notes                                  |
|------------------------------|-----------------------------|----------------------------------------|
| Summarise, extract           | `gemma3:e4b`                | Fast, strong instruction-following     |
| Reason, route, lint, claims  | `qwen3:14b`                 | Deep reasoning, thinking mode          |
| Embeddings                   | `nomic-embed-text:latest`   | 274 MB, MTEB-strong                    |
| Vision (image captions)      | `llava:7b`                  | Optional, when `ingest_caption_images` |

All served via Ollama at `OLLAMA_HOST` (default `http://localhost:11434`).

---

## Quickstart

```bash
git clone https://github.com/krishddd/llm-wiki.git
cd llm-wiki
pip install -r requirements.txt
cp .env.example .env  # set OLLAMA_HOST etc.

# Pull required models
ollama pull qwen3:14b
ollama pull gemma3:e4b
ollama pull nomic-embed-text

# Run the API
uvicorn src.api:app --reload --port 8000
```

Ingest a doc, then ask a question:

```bash
curl -F file=@paper.pdf http://localhost:8000/ingest
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"q": "What did the paper conclude about transformer scaling?"}'
```

Run the agent over MCP:

```bash
python -m src.mcp_server   # exposes ingest / query / lint as MCP tools
```

Trigger a job manually:

```bash
curl -X POST http://localhost:8000/admin/run/promote_episodic
```

---

## Project structure

```
src/
├── api.py                 FastAPI endpoints
├── ingest.py              Multi-format ingest pipeline
├── query.py               Hybrid retrieval + reflective synthesis + save-back
├── lint.py                Health check + auto-fix
├── graph.py               Bi-temporal knowledge graph (SQLite-backed)
├── llm.py                 Async Ollama client (cached embeddings)
├── config.py              pydantic-settings — all knobs
├── logging_config.py      JSON logs + audit channel
├── scheduler.py           APScheduler + JOB_REGISTRY
├── mcp_server.py          Agent-facing MCP wrapper
├── search/                BM25, dense, RRF, FlashRank, MMR, multi-query, intent
├── synth/                 Answer blocks, per-claim confidence, reflect, followups
├── loaders/               Multi-format (PDF, DOCX, PPTX, XLSX, HTML, MD, TXT)
└── wiki/                  Page store, episodic, promote, procedures,
                           reconciler, lifecycle, contradiction_resolver,
                           entity_pages, index_md, log_md

wiki/
├── index.md               Auto-regenerated table of contents
├── log.md                 Append-only operation log
├── sources/               Semantic tier — primary citable content
├── entities/              Semantic tier — auto-generated entity pages
├── procedures/            Procedural tier — recurring patterns
├── episodic/<date>.md     Episodic tier — append-only daily logs
├── archive/               Pages auto-moved here by lint auto-fix
├── review/                Confidence-gated drafts awaiting human accept
│   └── edits/             Reconciler-staged edit proposals
└── raw/                   Immutable source documents

data/
├── graph.db               SQLite: entities, relations, facts, page_access
├── procedures.db          SQLite: recurring query patterns
├── bm25.pkl               BM25 index
└── chroma/                ChromaDB persistence (or numpy fallback)

logs/
├── app.log                Rotating JSON, all events
└── audit.log              Filtered audit channel
```

---

## Page frontmatter convention

```yaml
---
title: "Page Title"
kind: source | entity | synthesis | promoted | crystallized | procedure
source: "wiki/raw/file.pdf" | "query-save-back" | "episodic-promotion"
ingested: 2026-05-01
confidence: 0.87
confidence_reason: "..."
tags: [concept, person, org]
entity_refs: ["Entity A", "Entity B"]
context_preamble: "..."     # Anthropic Contextual Retrieval
has_tables: true
has_images: false
element_counts: {text: 14, heading: 6, table: 3, image: 0, code: 0}
evolved_by:                 # populated when reconciler edits this page
  - {source: "Foo Doc", action: "refine", date: 2026-05-15}
correlation_ids: [COR-...]  # crystallized / promoted only
---
```

---

## CI & local development

GitHub Actions runs ruff, mypy, pytest (Ollama mocked), and a Docker build on
every push to `main`. Strict ruff config lives in `pyproject.toml`. The
integration suite (`workflows/integration.yml`) is gated behind a manually
triggered `workflow_dispatch` plus a `REMOTE_OLLAMA_HOST` secret, so day-to-day
CI never depends on a live LLM.

---

## Status

Personal research project. Explores how far a local-first LLM-driven wiki
can self-organise without a human curator.

## License

MIT
