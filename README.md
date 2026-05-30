# llm-wiki

> A self-healing local knowledge base where an LLM compounds your documents
> across four memory tiers — working, episodic, semantic, and procedural.

`llm-wiki` ingests PDFs, DOCX, PPTX, XLSX, HTML, and Markdown into a persistent
Markdown wiki backed by a bi-temporal knowledge graph. Every query is
remembered, recurring topics auto-promote to semantic pages, and repeated query
patterns crystallise into reusable procedures.

See [`CLAUDE.md`](./CLAUDE.md) for the full schema, memory tiers, and workflows.

## Features

- **Hybrid retrieval** — BM25 + dense + RRF + FlashRank + graph 2-hop + MMR,
  with multi-query paraphrase and HyDE seeding.
- **Bi-temporal facts** — every claim carries `ingested_at`, `valid_from`,
  `valid_to`, `superseded_by`, and `last_reinforced`. Effective confidence
  decays per an Ebbinghaus curve.
- **Auto-promotion** — episodic events that recur ≥ 3 times across ≥ 14 days
  become semantic pages.
- **Procedural memory** — query patterns recurring ≥ 5 times crystallise into
  procedures backed by SQLite + Markdown.
- **Auto-lint** — qwen3 scans pages weekly, archives stale low-confidence
  ones, rebuilds entity pages, fixes orphans.
- **Contradiction handling** — composite-score auto-resolver, with sub-margin
  cases surfaced for human review.
- **FastAPI + MCP server** — both REST and Model Context Protocol access.

## Tech stack

Python · FastAPI · Ollama (`qwen3:14b`, `gemma`, `nomic-embed-text`) ·
ChromaDB · APScheduler · SQLite

## Quickstart

```bash
git clone https://github.com/krishddd/llm-wiki.git
cd llm-wiki
pip install -r requirements.txt
cp .env.example .env  # set OLLAMA_HOST

# Pull required models
ollama pull qwen3:14b
ollama pull gemma3:e4b
ollama pull nomic-embed-text

# Run the API
uvicorn src.api:app --reload --port 8000
```

Ingest a document and ask a question:

```bash
curl -F file=@paper.pdf http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"q": "What did the paper conclude?"}'
```

## Project structure

```
src/
  api.py                FastAPI endpoints
  ingest.py             Multi-format ingest pipeline
  query.py              Hybrid retrieval + reflective synthesis
  graph.py              Bi-temporal knowledge graph
  scheduler.py          APScheduler jobs (decay / promote / lint / procedures)
  mcp_server.py         Model Context Protocol wrapper
  search/, synth/, loaders/, wiki/
wiki/                   Generated Markdown across four memory tiers
data/                   graph.db, procedures.db, bm25.pkl, chroma/
```

## Status

Personal research project — explores how far a local-first LLM-driven wiki
can self-organise without a human curator.

## License

MIT
