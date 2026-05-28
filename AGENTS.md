# AGENTS.md — LLM-Wiki agent surface

> Agent-facing companion to `CLAUDE.md`. Lists the tools, resources, and
> conventions an MCP-aware agent (Claude Code, Cursor, Codex, etc.) must
> follow when driving this wiki. Loaded by `src/mcp_server.py`.

## Quick start for an agent

1. Call **`context_start(days=7, top_pages=5)`** at the beginning of every new session.
   Returns recent episodic timeline + top accessed pages + open contradictions.
2. To answer a user question: call **`query_wiki(question, top_k=5, save_back=true)`**.
   The response carries numbered citations, per-claim confidences, and follow-up suggestions.
3. After a chain of related queries, call **`session_crystallize(correlation_ids=[...])`** to
   persist the exploration as one durable wiki page.
4. To add a document: call **`ingest_file(path)`** with an absolute path on the host.
5. Periodically (or when results feel stale) call **`run_admin_job("lint_autofix")`**
   or **`run_admin_job("decay_sweep")`** — same code paths as the in-process scheduler.

## Tool catalogue (MCP)

| Tool | Purpose |
|---|---|
| `query_wiki(question, top_k=5, save_back=true)` | Hybrid retrieval + reflective synthesis. Returns answer with `[1] [2]`-style citations, blocks, follow-ups, intent, retrieval quality. |
| `ingest_file(path)` | Multi-format ingest (PDF/DOCX/HTML/PPTX/XLSX/CSV/MD/TXT). Confidence-gated write. |
| `lint_wiki()` | Read-only health report. |
| `list_entities(limit=50)` | Canonical entities by backlink count. |
| `session_crystallize(correlation_ids, days=14)` | Distil an exploration thread into one wiki page. |
| `context_start(days=7, top_pages=5)` | Session-start briefing. |
| `run_admin_job(job_name)` | Manually trigger one of: `decay_sweep`, `episodic_prune`, `promote_episodic`, `lint_autofix`, `detect_procedures`, `lint`. |

## Resources (MCP)

| Resource URI | Content |
|---|---|
| `wiki://index` | The current `wiki/index.md` (master catalogue, regenerated deterministically per ingest). |

## REST endpoints (when used directly without MCP)

```
GET  /health
POST /ingest                       (multipart files=)
POST /query                        (json body)
GET  /lint                         (read-only)
POST /lint                         (with auto_fix=true to repair)
GET  /entities                     (?limit=200)
GET  /entities/{entity_id}
GET  /facts/{entity_name}          (?history=true; carries effective_confidence)
GET  /episodic                     (?days=3)
GET  /context/start                (?days=7&top_pages=5)
POST /session/crystallize          {correlation_ids:[…], days:14}
GET  /review                       (staged low-confidence pages)
GET  /review/edits                 (reconciler edit-proposals)
POST /review/{id}/accept | reject
GET  /admin/jobs                   (registered scheduled jobs + next-fire times)
POST /admin/run/{job_name}         (manual job trigger)
GET  /admin/contradictions         (unresolved fact-pair contradictions)
POST /admin/contradictions/resolve  (?fact_a, ?fact_b, ?force_winner=)
GET  /wiki/index
```

## Conventions an agent MUST follow

### Citations

- Inline numbered citations only — `[1] [2]`. The numbers map to the response's `citations[]` array.
- Never fabricate a `[N]` for a page not in `retrieved_pages`. The grounding check will reject the answer.
- For per-claim confidence (when emitted internally as `[Title]^0.92`), the post-processor strips the `^0.NN` before the user sees it. Do not pre-strip.

### When to ingest vs. when to crystallize

- **Ingest** = a new external document. Always uses `ingest_file`.
- **Crystallize** = consolidate a chain of YOUR queries into a wiki page (your exploration becomes a source for next time). Always uses `session_crystallize`.
- **Save-back** = automatic, no agent action — high-confidence query answers are filed back as `wiki/sources/synthesis-*.md`.

### Reading effective confidence

Every fact returned by `/facts/{name}` has both `confidence` (stored) and `effective_confidence` (decayed). Use `effective_confidence` when explaining to a user how reliable a claim is. Use `confidence` when comparing two claims for the auto-resolver.

### Contradictions

If you detect that two of your sources disagree, do NOT manually rewrite either page. Instead:
1. Surface the contradiction in your answer (cite both).
2. The auto-resolver will run on next ingest of any related source.
3. Or call `POST /admin/contradictions/resolve?fact_a=X&fact_b=Y` to trigger composite scoring.
4. If you have ground truth, pass `force_winner=<id>` to skip the score.

### Privacy

Never include API keys, tokens, passwords, or PII in queries or in pages you crystallize. The redactor module is not yet implemented — the agent is the first line of defence.

### Idempotency

- All schedule-driven jobs are idempotent. Calling `run_admin_job("lint_autofix")` twice in a row is safe.
- `ingest_file` is *not* fully idempotent — re-ingesting the same file creates a second page (the title changes only if the source changes). Prefer human review.

### When in doubt, read CLAUDE.md

It carries the human-facing depth: the four memory tiers, decay/reinforcement
policy, supersession lifecycle, audit-event taxonomy, and the directory layout.
This file (`AGENTS.md`) is intentionally minimal — tool catalogue + behavioural
contract.

---

*v2 last updated: 2026-05-01.*
