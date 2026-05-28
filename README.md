# LLM-Wiki PoC — Advanced Retrieval & GraphRAG

A FastAPI service that ingests documents into a persistent, compounding Markdown wiki and answers
questions via hybrid BM25 + dense + FlashRank + graph-expanded retrieval, backed by a remote Ollama
instance running `qwen3:14b` / `gemma3:e4b` / `nomic-embed-text` / `llama3.2` / `llava:7b`.

## Quickstart

```bash
cp .env.example .env     # set OLLAMA_HOST to your remote instance
make install
make test                # unit tests (Ollama mocked)
make dev                 # uvicorn http://localhost:8000
```

Docker:

```bash
make up                  # docker compose up -d --build
make health              # curl :8000/health
```

## REST API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Ollama reachability + required-model check |
| POST | `/ingest` | multipart upload of one-or-more files → per-file result |
| POST | `/query` | structured JSON answer (summary, key_points, citations, entities, confidence) |
| GET | `/lint` | Qwen health-check report (orphans, stale, contradictions, ...) |
| GET | `/review` | list low-confidence staged pages |
| POST | `/review/{id}/accept` | promote staged page to `wiki/sources/` |
| POST | `/review/{id}/reject` | delete staged page |
| GET | `/wiki/index` | serve `wiki/index.md` |

Every response has an `X-Correlation-ID` header and `correlation_id` body field. The same id is
written into every JSON log line (`logs/app.log` + `logs/audit.log`), so `grep COR-<id> logs/app.log`
reconstructs the full API → route → retrieve → graph-expand → LLM synth chain.

## Retrieval Pipeline (Days 8–12)

```
query
 ├─ BM25 (rank-bm25, pickled)        ┐
 ├─ Dense (ChromaDB + nomic-embed)   ┴─ asyncio.gather
 ├─ RRF fusion → top 40
 ├─ FlashRank cross-encoder → top 5
 ├─ Graph expand (entities within 2 hops, capped +10) → rerank again → top 5
 └─ Qwen3 synth → structured JSON (answer + summary + key_points + citations + entities + confidence)
```

## Ingest Pipeline

1. Multi-file upload → save under `wiki/raw/`.
2. Per file: loader (PDF via pypdf, MD/HTML passthrough) → chunk.
3. Gemma summarise+extract per chunk — **throttled by `asyncio.Semaphore(MAX_CONCURRENT_LLM_REQ)`**
   so remote Ollama's same-model queue doesn't timeout-cascade.
4. Qwen merge partial summaries → confidence score.
5. Write page — `≥ CONFIDENCE_THRESHOLD (0.60)` → `wiki/sources/`, else `wiki/review/`.
6. Entity canonicalization via `thefuzz.token_sort_ratio ≥ 90` → upsert into SQLite graph.
7. BM25 + Chroma upserts.
8. Regenerate `wiki/index.md`, append `wiki/log.md`.

## Remote Ollama

- Dev machine: run a tiny local Ollama (`llama3.2` + `nomic-embed-text`) or the in-memory fake used
  by the unit tests.
- CI: `.github/workflows/integration.yml` points at `secrets.REMOTE_OLLAMA_HOST`. Gated on
  `workflow_dispatch`, nightly cron, and pushes to `main` — not on every PR.

## CI/CD

- `.github/workflows/ci.yml` — ruff + mypy + `pytest -m "not integration"` + `docker build`.
- `.github/workflows/integration.yml` — spins up the stack against the remote Ollama and runs
  `pytest -m integration`.

## Layout

See `CLAUDE.md` in the parent directory for the full schema, frontmatter conventions, and audit
event types.
