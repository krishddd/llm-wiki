"""FastAPI app — multi-doc ingest, structured Q&A, lint, review workflow.

Correlation-ID middleware mints a UUID per request, stores it in a contextvars.ContextVar,
and returns it both as `X-Correlation-ID` header and inside response payloads. All downstream
logs pick it up automatically via the JsonFormatter.
"""
from __future__ import annotations

import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .lint import lint_wiki
from .llm import close_client, get_client
from .logging_config import (
    audit,
    correlation_id_ctx,
    new_correlation_id,
    setup_logging,
    timed_ms,
)
from .query import QueryEngine
from .search.bm25_index import BM25Index
from .search.dense_index import DenseIndex
from .wiki.entity_pages import rebuild_entity_pages
from .wiki.index_md import rebuild_index
from .wiki.log_md import append_log
from .wiki.pages import PageStore, page_id_from_path, read_page, write_page

log = logging.getLogger(__name__)

app = FastAPI(title="LLM-Wiki PoC", version="0.1.0")

import contextlib

from fastapi.staticfiles import StaticFiles

app.mount("/dashboard", StaticFiles(directory="src/static", html=True), name="static")


class _State:
    bm25: BM25Index
    dense: DenseIndex
    graph: KnowledgeGraph
    page_store: PageStore
    ingestor: Ingestor
    query_engine: QueryEngine
    procedures: Any | None = None
    scheduler: Any | None = None


state = _State()


@app.on_event("startup")
async def _startup() -> None:
    s = get_settings()
    setup_logging(s.logs_dir)
    s.wiki_dir.mkdir(parents=True, exist_ok=True)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("sources", "entities", "review", "procedures", "episodic", "archive"):
        (s.wiki_dir / sub).mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    state.bm25 = BM25Index(s.data_dir / "bm25.pkl")
    state.dense = DenseIndex(s.data_dir / "chroma", embed_fn=client.embed)
    state.graph = KnowledgeGraph(s.data_dir / "graph.db")
    state.page_store = PageStore(s.wiki_dir)
    state.ingestor = Ingestor(client=client, graph=state.graph, bm25=state.bm25, dense=state.dense)
    state.query_engine = QueryEngine(
        bm25=state.bm25, dense=state.dense, page_store=state.page_store, graph=state.graph, client=client
    )
    # Phase C2: procedural memory store (own SQLite file).
    from .wiki.procedures import ProcedureStore
    state.procedures = ProcedureStore(s.data_dir / "procedures.db")
    state.query_engine.procedures = state.procedures   # let QueryEngine record patterns

    # Phase D: in-process APScheduler for background upkeep jobs.
    if getattr(s, "scheduler_enabled", True):
        try:
            from .scheduler import make_scheduler
            state.scheduler = make_scheduler(state)
            if state.scheduler is not None:
                state.scheduler.start()
                log.info(
                    "scheduler started",
                    extra={"metadata": {"jobs": [j.id for j in state.scheduler.get_jobs()]}},
                )
        except Exception as e:
            log.warning("scheduler startup failed",
                        extra={"metadata": {"error": str(e)[:200]}})
            state.scheduler = None

    log.info("startup complete", extra={"metadata": {"wiki_dir": str(s.wiki_dir)}})


@app.on_event("shutdown")
async def _shutdown() -> None:
    if getattr(state, "scheduler", None) is not None:
        with contextlib.suppress(Exception):
            state.scheduler.shutdown(wait=False)
    if getattr(state, "procedures", None) is not None:
        with contextlib.suppress(Exception):
            state.procedures.close()
    await close_client()
    state.graph.close()


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    cid = request.headers.get("x-correlation-id") or new_correlation_id()
    token = correlation_id_ctx.set(cid)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"metadata": {"path": request.url.path}})
            response = JSONResponse({"error": "internal error", "correlation_id": cid}, status_code=500)
        response.headers["X-Correlation-ID"] = cid
        log.info(
            "request",
            extra={
                "metadata": {"path": request.url.path, "method": request.method, "status": response.status_code},
                "duration_ms": timed_ms(start),
            },
        )
        return response
    finally:
        correlation_id_ctx.reset(token)


# ───── Schemas ─────

class QueryBody(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)
    graph_expand: bool = True
    use_hyde: bool = True
    decompose: bool = True
    save_back: bool = True


class CitationExcerptOut(BaseModel):
    kind: str           # "table" | "image" | "code"
    content: str        # ready-to-render Markdown
    meta: dict = {}


class CitationOut(BaseModel):
    page: str
    title: str
    snippet: str
    has_tables: bool = False
    has_images: bool = False
    has_code: bool = False
    excerpts: list[CitationExcerptOut] = []


class AnswerBlockOut(BaseModel):
    kind: str            # "heading" | "text" | "list" | "table" | "code" | "math" | "quote" | "callout"
    content: str
    meta: dict = {}


class QueryResponse(BaseModel):
    answer: str                    # numbered-citation Markdown — drop-in for any renderer
    summary: str
    key_points: list[str]
    blocks: list[AnswerBlockOut] = []
    follow_up_questions: list[str] = []
    citations: list[CitationOut]
    entities: list[str]
    confidence: float
    correlation_id: str
    retrieved_pages: list[str]
    sub_queries: list[str] = []
    grounded: bool = True
    saved_page: str | None = None
    retrieval_quality: str = "correct"   # CRAG verdict: correct | ambiguous | incorrect
    intent: str = "synthesis"            # factual | multi_hop | synthesis | exhaustive
    quality_score: float = 1.0           # reflection critique 0-1
    quality_issues: list[str] = []
    per_claim_confidences: list[dict] = []   # [{"citation": "...", "confidence": 0.92}, ...]


# ───── Endpoints ─────

@app.get("/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    client = get_client()
    try:
        models = await client.list_models()
        reachable = True
    except Exception as e:
        models, reachable = [], False
        log.warning("ollama unreachable", extra={"metadata": {"error": str(e)}})
    required = s.required_models()
    missing = [m for m in required if m not in models]
    return {
        "ok": reachable and not missing,
        "ollama_reachable": reachable,
        "ollama_host": s.ollama_host,
        "models_available": models,
        "models_required": required,
        "models_missing": missing,
        "correlation_id": correlation_id_ctx.get(),
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(400, "no files provided")
    s = get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for f in files:
        if not f.filename:
            continue
        target = s.raw_dir / Path(f.filename).name
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(target)
    results = await state.ingestor.ingest_many(saved_paths)
    return {
        "batch_id": new_correlation_id(),
        "correlation_id": correlation_id_ctx.get(),
        "count": len(results),
        "results": [r.__dict__ for r in results],
    }


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryBody) -> QueryResponse:
    s = get_settings()
    result = await state.query_engine.answer(
        body.question,
        top_k=body.top_k,
        graph_expand=body.graph_expand,
        use_hyde=body.use_hyde,
        decompose=body.decompose,
        save_back=body.save_back,
    )
    # Episodic log of every query
    if s.episodic_logging:
        try:
            from .wiki.episodic import append_episode
            append_episode(
                s.wiki_dir,
                kind="query",
                title=body.question[:120],
                body=(result.summary or "")[:500],
                correlation_id=correlation_id_ctx.get(),
                metadata={
                    "intent": getattr(result, "intent", "synthesis"),
                    "confidence": f"{result.confidence:.2f}",
                    "grounded": str(getattr(result, "grounded", True)),
                    "retrieval_quality": getattr(result, "retrieval_quality", "correct"),
                    "n_citations": len(result.citations),
                },
            )
        except Exception as e:
            log.debug("episodic logging failed", extra={"metadata": {"error": str(e)[:120]}})
    cit_out: list[CitationOut] = []
    for c in result.citations:
        excerpts_out = [
            CitationExcerptOut(kind=e.kind, content=e.content, meta=getattr(e, "meta", {}) or {})
            for e in (getattr(c, "excerpts", None) or [])
        ]
        cit_out.append(
            CitationOut(
                page=c.page,
                title=c.title,
                snippet=c.snippet,
                has_tables=getattr(c, "has_tables", False),
                has_images=getattr(c, "has_images", False),
                has_code=getattr(c, "has_code", False),
                excerpts=excerpts_out,
            )
        )
    blocks_out = [
        AnswerBlockOut(kind=b.kind, content=b.content, meta=getattr(b, "meta", {}) or {})
        for b in (getattr(result, "blocks", None) or [])
    ]
    return QueryResponse(
        answer=result.answer,
        summary=result.summary,
        key_points=result.key_points,
        blocks=blocks_out,
        follow_up_questions=getattr(result, "follow_up_questions", []) or [],
        citations=cit_out,
        entities=result.entities,
        confidence=result.confidence,
        correlation_id=correlation_id_ctx.get(),
        retrieved_pages=result.retrieved_pages,
        sub_queries=getattr(result, "sub_queries", []) or [],
        grounded=getattr(result, "grounded", True),
        retrieval_quality=getattr(result, "retrieval_quality", "correct"),
        intent=getattr(result, "intent", "synthesis"),
        quality_score=getattr(result, "quality_score", 1.0),
        quality_issues=getattr(result, "quality_issues", []) or [],
        per_claim_confidences=getattr(result, "per_claim_confidences", []) or [],
        saved_page=getattr(result, "saved_page", None),
    )


@app.get("/lint")
async def lint() -> dict[str, Any]:
    """Read-only lint report. State-mutating runs use POST /lint."""
    report = await lint_wiki(
        page_store=state.page_store,
        graph=state.graph,
        bm25=state.bm25,
        dense=state.dense,
        auto_fix=False,
    )
    report["correlation_id"] = correlation_id_ctx.get()
    return report


@app.post("/lint")
async def lint_post(auto_fix: bool = False) -> dict[str, Any]:
    """Lint with optional auto-repair. Same job is also reachable via
    `POST /admin/run/lint_autofix` when called from a scheduler client."""
    report = await lint_wiki(
        page_store=state.page_store,
        graph=state.graph,
        bm25=state.bm25,
        dense=state.dense,
        auto_fix=bool(auto_fix),
    )
    report["correlation_id"] = correlation_id_ctx.get()
    return report


@app.get("/review")
async def list_review() -> dict[str, Any]:
    s = get_settings()
    review_dir = s.wiki_dir / "review"
    items = []
    for p in sorted(review_dir.glob("*.md")):
        page = read_page(p)
        fm = page.frontmatter or {}
        items.append({
            "id": p.stem,
            "path": str(p).replace("\\", "/"),
            "title": fm.get("title"),
            "confidence": fm.get("confidence"),
            "source": fm.get("source"),
        })
    return {"count": len(items), "items": items}


@app.post("/review/{review_id}/accept")
async def accept_review(review_id: str) -> dict[str, Any]:
    s = get_settings()
    src = s.wiki_dir / "review" / f"{review_id}.md"
    if not src.exists():
        raise HTTPException(404, f"no review page {review_id}")
    dst = s.wiki_dir / "sources" / f"{review_id}.md"
    page = read_page(src)
    page.path = dst
    write_page(page)
    src.unlink()
    pid = page_id_from_path(dst, s.wiki_dir)
    await state.bm25.upsert(pid, f"{page.frontmatter.get('title', review_id)}\n{page.body}")
    await state.dense.upsert(pid, f"{page.frontmatter.get('title', review_id)}\n{page.body}")
    rebuild_index(s.wiki_dir)
    try:
        rebuild_entity_pages(state.graph, s.wiki_dir)
    except Exception as e:
        log.warning("entity-page rebuild failed", extra={"metadata": {"error": str(e)[:200]}})
    append_log(s.wiki_dir, "review-accept", review_id)
    audit(log, "WIKI_REVIEW_ACCEPT", str(dst))
    return {"ok": True, "page_path": str(dst).replace("\\", "/")}


@app.post("/review/{review_id}/reject")
async def reject_review(review_id: str) -> dict[str, Any]:
    s = get_settings()
    src = s.wiki_dir / "review" / f"{review_id}.md"
    if not src.exists():
        raise HTTPException(404, f"no review page {review_id}")
    src.unlink()
    append_log(s.wiki_dir, "review-reject", review_id)
    audit(log, "WIKI_REVIEW_REJECT", str(src))
    return {"ok": True}


@app.get("/wiki/index")
async def wiki_index() -> dict[str, str]:
    s = get_settings()
    idx = s.wiki_dir / "index.md"
    text = idx.read_text(encoding="utf-8") if idx.exists() else ""
    return {"content": text}


@app.get("/facts/{entity_name}")
async def facts_for_entity(entity_name: str, history: bool = False) -> dict[str, Any]:
    """Bi-temporal facts about an entity. Default: only currently-active facts.
    `history=true` returns the full audit trail including superseded facts.

    Each fact carries both the stored `confidence` AND `effective_confidence`
    (decayed via Ebbinghaus curve based on `last_reinforced` / `ingested_at`).
    """
    if history:
        items = await state.graph.history_for(entity_name)
    else:
        items = await state.graph.active_facts_for(entity_name)

    # Phase B4: compute effective_confidence on the fly without mutating storage.
    s = get_settings()
    if getattr(s, "lifecycle_enabled", True):
        from .wiki.lifecycle import effective_confidence
        for it in items:
            seed = it.get("last_reinforced") or it.get("ingested_at")
            it["effective_confidence"] = round(
                effective_confidence(
                    stored=float(it.get("confidence") or 0.0),
                    last_reinforced=seed,
                    half_life_days=getattr(s, "decay_half_life_days", 90.0),
                ),
                3,
            )
    return {"entity": entity_name, "history": history, "count": len(items), "items": items}


CRYSTALLIZE_SYSTEM = (
    "You are distilling a chain of related research/exploration episodes into ONE durable wiki page. "
    "Each episode below is one Q&A or ingest. Produce: (1) a short title, (2) a 1-sentence TL;DR, "
    "(3) the question evolution (how the focus shifted), (4) what was found (key findings, with "
    "[[page-stem|Title]] wiki-links to cited pages), (5) lessons / open questions. "
    "Reply ONLY JSON: "
    '{"title":"…","summary":"…","question_evolution":["…"],"findings":["…"],'
    '"lessons":["…"],"open_questions":["…"],"cited_pages":["page-id"]}'
)


class CrystallizeBody(BaseModel):
    correlation_ids: list[str]
    days: int = 14
    title_hint: str | None = None


@app.post("/session/crystallize")
async def session_crystallize(body: CrystallizeBody) -> dict[str, Any]:
    """Phase F1: distil an exploration thread (matched by correlation IDs) into
    one wiki page filed under wiki/sources/crystallized-<slug>.md."""
    s = get_settings()
    import json as _json
    import re as _re

    from .llm import get_client
    from .wiki.episodic import read_episodes
    from .wiki.synth_page import SynthesisPageInputs, write_synthesis_page

    episodes = read_episodes(
        s.wiki_dir,
        correlation_ids=body.correlation_ids,
        days=max(1, body.days),
    )
    if not episodes:
        return {"ok": False, "error": "no episodes matched the given correlation_ids"}

    # Build the prompt from episodes.
    lines = []
    for e in episodes[:30]:
        lines.append(f"### [{e['date']} {e['time']}] {e['kind']} — {e['title']}")
        lines.append(e["body"][:1500])
        lines.append("")
    prompt = "EXPLORATION EPISODES:\n\n" + "\n".join(lines) + "\n\nDistil the thread."
    client = get_client()
    try:
        raw = await client.qwen(prompt, system=CRYSTALLIZE_SYSTEM, temperature=0.2)
    except Exception as e:
        return {"ok": False, "error": f"crystallize LLM call failed: {str(e)[:200]}"}

    # Lenient JSON extraction (mirrors other modules).
    s_clean = _re.sub(r"^```(?:json)?\n?", "", (raw or "").strip())
    s_clean = _re.sub(r"\n?```$", "", s_clean)
    m = _re.search(r"\{.*\}", s_clean, _re.DOTALL)
    parsed: dict = {}
    if m:
        try:
            parsed = _json.loads(m.group(0))
        except _json.JSONDecodeError:
            parsed = {}

    title = (body.title_hint or parsed.get("title") or
             (episodes[0]["title"][:120] if episodes else "Crystallized session"))[:120]

    body_lines = [f"# {title}", ""]
    if parsed.get("summary"):
        body_lines.append(f"**TL;DR:** {parsed['summary']}")
        body_lines.append("")
    if parsed.get("question_evolution"):
        body_lines.append("## Question evolution")
        body_lines.append("")
        for q in parsed["question_evolution"]:
            body_lines.append(f"- {q}")
        body_lines.append("")
    if parsed.get("findings"):
        body_lines.append("## Findings")
        body_lines.append("")
        for f in parsed["findings"]:
            body_lines.append(f"- {f}")
        body_lines.append("")
    if parsed.get("lessons"):
        body_lines.append("## Lessons")
        body_lines.append("")
        for l in parsed["lessons"]:
            body_lines.append(f"- {l}")
        body_lines.append("")
    if parsed.get("open_questions"):
        body_lines.append("## Open questions")
        body_lines.append("")
        for q in parsed["open_questions"]:
            body_lines.append(f"- {q}")
        body_lines.append("")
    if parsed.get("cited_pages"):
        body_lines.append("## Sources")
        body_lines.append("")
        for c in parsed["cited_pages"]:
            stem = Path(str(c)).stem
            body_lines.append(f"- [[{stem}]]")

    fm = {
        "title": title,
        "kind": "crystallized",
        "source": "session-crystallize",
        "correlation_ids": list(body.correlation_ids)[:50],
        "episode_count": len(episodes),
        "confidence": 0.7,
        "created": datetime.now(UTC).date().isoformat(),
    }
    pid = await write_synthesis_page(
        wiki_dir=s.wiki_dir,
        bm25=state.bm25,
        dense=state.dense,
        inputs=SynthesisPageInputs(
            title=title,
            body="\n".join(body_lines),
            frontmatter=fm,
            page_kind="crystallized",
        ),
    )
    return {
        "ok": pid is not None,
        "page_id": pid,
        "episodes_used": len(episodes),
        "correlation_id": correlation_id_ctx.get(),
    }


@app.get("/context/start")
async def context_start(days: int = 7, top_pages: int = 5) -> dict[str, Any]:
    """Phase F2: Session-start briefing — recent episodic + most-accessed pages
    + open contradictions. The endpoint an MCP client / agent loads at session start."""
    from .wiki.contradiction_resolver import list_unresolved_contradictions
    from .wiki.episodic import list_recent_episodes
    s = get_settings()

    recent = list_recent_episodes(s.wiki_dir, days=max(1, days))

    # Top-N most-accessed pages (Phase B reinforcement counts).
    # Acquire the graph lock — page_access is shared with the reinforcement
    # writer in src/wiki/lifecycle.py:mark_accessed.
    top: list[dict] = []
    try:
        async with state.graph._lock:
            cur = state.graph._conn.cursor()
            cur.execute(
                "SELECT page_id, access_count, last_accessed, last_reinforced "
                "FROM page_access ORDER BY access_count DESC LIMIT ?",
                (max(1, top_pages),),
            )
            rows = cur.fetchall()
        for row in rows:
            top.append({
                "page_id": row[0],
                "access_count": int(row[1] or 0),
                "last_accessed": row[2],
                "last_reinforced": row[3],
            })
    except Exception as e:
        log.debug("context/start: page_access query failed",
                  extra={"metadata": {"error": str(e)[:200]}})

    open_contradictions = await list_unresolved_contradictions(state.graph, limit=10)

    return {
        "days": days,
        "recent_episodes": recent[:50],
        "top_pages": top,
        "open_contradictions": open_contradictions,
        "correlation_id": correlation_id_ctx.get(),
    }


@app.get("/admin/contradictions")
async def admin_contradictions(limit: int = 50) -> dict[str, Any]:
    """List unresolved contradictions (active fact pairs with same subject+predicate
    but different objects). Useful when the auto-resolver decided the margin
    was too small."""
    from .wiki.contradiction_resolver import list_unresolved_contradictions
    items = await list_unresolved_contradictions(state.graph, limit=limit)
    return {"count": len(items), "items": items}


@app.post("/admin/contradictions/resolve")
async def admin_contradictions_resolve(
    fact_a: int, fact_b: int, force_winner: int | None = None
) -> dict[str, Any]:
    """Auto-resolve a specific contradiction. If `force_winner` is provided,
    skip the score and supersede the other fact directly."""
    s = get_settings()
    if force_winner is not None:
        loser = fact_b if force_winner == fact_a else fact_a
        from datetime import datetime
        today = datetime.now(UTC).date().isoformat()
        try:
            await state.graph.supersede_fact(
                old_fact_id=loser, new_fact_id=force_winner, valid_to=today,
            )
            return {"resolved": True, "winner": force_winner, "loser": loser,
                    "reason": "human-forced"}
        except Exception as e:
            return {"resolved": False, "error": str(e)[:300]}
    from .wiki.contradiction_resolver import resolve_contradiction
    return await resolve_contradiction(
        state.graph, fact_a, fact_b, wiki_dir=s.wiki_dir,
    )


@app.get("/admin/jobs")
async def list_jobs() -> dict[str, Any]:
    """List registered scheduled jobs and their next-fire times."""
    from .scheduler import JOB_REGISTRY
    available = sorted(JOB_REGISTRY.keys())
    scheduled = []
    if getattr(state, "scheduler", None) is not None:
        for j in state.scheduler.get_jobs():
            scheduled.append({
                "id": j.id, "name": j.name,
                "next_run": str(j.next_run_time) if j.next_run_time else None,
            })
    return {
        "available_for_manual_run": available,
        "scheduled": scheduled,
    }


@app.post("/admin/run/{job_name}")
async def run_admin_job(job_name: str) -> dict[str, Any]:
    """Manually trigger a registered job — same code path as the scheduler."""
    from .scheduler import run_job_now
    return await run_job_now(state, job_name)


@app.get("/episodic")
async def episodic(days: int = 3) -> dict[str, Any]:
    """Recent episodic entries (queries / ingests / lints) from the last `days` days."""
    from .wiki.episodic import list_recent_episodes
    s = get_settings()
    items = list_recent_episodes(s.wiki_dir, days=days)
    return {"days": days, "count": len(items), "items": items}


@app.get("/review/edits")
async def list_edit_proposals() -> dict[str, Any]:
    """Staged reconciler edit-proposals awaiting human review."""
    s = get_settings()
    edits_dir = s.wiki_dir / "review" / "edits"
    items = []
    if edits_dir.exists():
        for p in sorted(edits_dir.glob("*.md")):
            try:
                page = read_page(p)
                fm = page.frontmatter or {}
                items.append({
                    "id": p.stem,
                    "path": str(p).replace("\\", "/"),
                    "target_page": fm.get("target_page"),
                    "new_source": fm.get("new_source"),
                    "action": fm.get("action"),
                    "confidence": fm.get("confidence"),
                })
            except Exception:
                continue
    return {"count": len(items), "items": items}


@app.get("/entities")
async def list_entities(limit: int = 200) -> dict[str, Any]:
    """List canonical entities with backlink counts. Reads straight from the graph DB."""
    async with state.graph._lock:
        cur = state.graph._conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.name, e.type,
                   (SELECT COUNT(DISTINCT pe.page_id)
                      FROM page_entities pe JOIN entities e2 ON pe.entity_id = e2.id
                      WHERE e2.id = e.id OR e2.canonical_id = e.id) AS backlinks
            FROM entities e
            WHERE e.canonical_id IS NULL
            ORDER BY backlinks DESC, e.name ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    items = [
        {"id": int(r[0]), "name": str(r[1]), "type": str(r[2]), "backlinks": int(r[3])}
        for r in rows
    ]
    return {"count": len(items), "items": items}


@app.get("/entities/{entity_id}")
async def entity_detail(entity_id: int) -> dict[str, Any]:
    async with state.graph._lock:
        cur = state.graph._conn.cursor()
        cur.execute("SELECT id, name, type, canonical_id FROM entities WHERE id = ?", (entity_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"no entity {entity_id}")
        canon_id = row[3] if row[3] is not None else row[0]
        cur.execute("SELECT name FROM entities WHERE canonical_id = ? ORDER BY name", (canon_id,))
        aliases = [str(r[0]) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT DISTINCT pe.page_id
            FROM page_entities pe JOIN entities e ON pe.entity_id = e.id
            WHERE e.id = ? OR e.canonical_id = ?
            """,
            (canon_id, canon_id),
        )
        backlinks = [str(r[0]) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT r.rel_type, e.name, e.type
            FROM relations r JOIN entities e ON r.dst = e.id
            WHERE r.src = ? OR r.src IN (SELECT id FROM entities WHERE canonical_id = ?)
            UNION
            SELECT r.rel_type, e.name, e.type
            FROM relations r JOIN entities e ON r.src = e.id
            WHERE r.dst = ? OR r.dst IN (SELECT id FROM entities WHERE canonical_id = ?)
            LIMIT 50
            """,
            (canon_id, canon_id, canon_id, canon_id),
        )
        relation_rows = cur.fetchall()
    relations = [{"rel_type": str(r[0]), "name": str(r[1]), "type": str(r[2])} for r in relation_rows]
    return {
        "id": int(canon_id),
        "name": str(row[1]),
        "type": str(row[2]),
        "aliases": aliases,
        "backlinks": backlinks,
        "backlink_count": len(backlinks),
        "relations": relations,
    }
