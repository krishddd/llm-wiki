"""FastAPI router for the agentic RAG layer.

Mounted alongside the existing routes in `api.py`. The existing `/query`
endpoint is untouched. This adds:

  POST /query/agentic   — iterative SCA-driven retrieval + synthesis
  GET  /agentic/health  — quick configuration peek
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .agentic_rag.config import get_agentic_settings
from .agentic_rag.orchestrator import QueryOrchestrator
from .logging_config import correlation_id_ctx

log = logging.getLogger(__name__)

router = APIRouter()


class AgenticQueryBody(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)
    graph_expand: bool = True
    save_back: bool = True
    force_agentic: bool = False
    force_simple: bool = False
    max_iterations: int | None = Field(None, ge=1, le=5)
    coverage_threshold: float | None = Field(None, ge=0.0, le=1.0)


def _serialize_result(result: Any) -> dict[str, Any]:
    """Reuse the same shape as the existing /query response, without re-coupling."""
    cits = []
    for c in (result.citations or []):
        excerpts = [
            {"kind": e.kind, "content": e.content, "meta": getattr(e, "meta", {}) or {}}
            for e in (getattr(c, "excerpts", None) or [])
        ]
        cits.append({
            "page": c.page,
            "title": c.title,
            "snippet": c.snippet,
            "has_tables": getattr(c, "has_tables", False),
            "has_images": getattr(c, "has_images", False),
            "has_code": getattr(c, "has_code", False),
            "excerpts": excerpts,
        })
    blocks = [
        {"kind": b.kind, "content": b.content, "meta": getattr(b, "meta", {}) or {}}
        for b in (getattr(result, "blocks", None) or [])
    ]
    return {
        "answer": result.answer,
        "summary": result.summary,
        "key_points": result.key_points,
        "blocks": blocks,
        "follow_up_questions": getattr(result, "follow_up_questions", []) or [],
        "citations": cits,
        "entities": result.entities,
        "confidence": result.confidence,
        "correlation_id": correlation_id_ctx.get(),
        "retrieved_pages": result.retrieved_pages,
        "sub_queries": getattr(result, "sub_queries", []) or [],
        "grounded": getattr(result, "grounded", True),
        "saved_page": getattr(result, "saved_page", None),
        "retrieval_quality": getattr(result, "retrieval_quality", "correct"),
        "intent": getattr(result, "intent", "synthesis"),
        "quality_score": getattr(result, "quality_score", 1.0),
        "quality_issues": getattr(result, "quality_issues", []) or [],
        "per_claim_confidences": getattr(result, "per_claim_confidences", []) or [],
    }


def get_state():
    """Lazy import of the api module's global state to avoid circular imports."""
    from . import api as api_mod
    return api_mod.state


@router.get("/agentic/health")
async def agentic_health() -> dict[str, Any]:
    s = get_agentic_settings()
    return {
        "enabled": s.agentic_enabled,
        "max_iterations": s.agentic_max_iterations,
        "coverage_threshold": s.agentic_coverage_threshold,
        "complexity_threshold": s.agentic_complexity_threshold,
        "fanout_concurrency": s.agentic_fanout_concurrency,
    }


@router.post("/query/agentic")
async def agentic_query(body: AgenticQueryBody) -> dict[str, Any]:
    state = get_state()
    engine = getattr(state, "query_engine", None)
    if engine is None:
        raise HTTPException(503, "query engine not initialised")

    settings = get_agentic_settings()
    if not settings.agentic_enabled and not body.force_agentic:
        raise HTTPException(503, "agentic layer disabled (AGENTIC_ENABLED=false)")

    orch = QueryOrchestrator(engine)
    try:
        result = await orch.process(
            body.question,
            top_k=body.top_k,
            graph_expand=body.graph_expand,
            save_back=body.save_back,
            force_agentic=body.force_agentic,
            force_simple=body.force_simple,
        )
    except Exception as e:
        log.exception("agentic query failed")
        raise HTTPException(500, f"agentic query failed: {e}") from e

    # Episodic log (best-effort, mirrors the existing /query handler).
    try:
        from .config import get_settings
        s = get_settings()
        if getattr(s, "episodic_logging", False):
            from .wiki.episodic import append_episode
            append_episode(
                s.wiki_dir,
                kind="query",
                title=("[agentic] " + body.question)[:120],
                body=(result.summary or "")[:500],
                correlation_id=correlation_id_ctx.get(),
                metadata={
                    "intent": getattr(result, "intent", "synthesis"),
                    "confidence": f"{result.confidence:.2f}",
                    "grounded": str(getattr(result, "grounded", True)),
                    "agentic": "true",
                    "n_citations": len(result.citations),
                },
            )
    except Exception as e:
        log.debug("agentic episodic logging failed",
                  extra={"metadata": {"error": str(e)[:120]}})

    return _serialize_result(result)
