"""Agentic iterative-retrieval loop.

Wraps the existing `QueryEngine`. On each iteration:
  1. Build context from all pages accumulated so far.
  2. Draft a quick answer.
  3. Ask the Sufficient Context Agent: is this enough?
  4. If YES → run the existing synthesis pipeline on the accumulated pages.
  5. If NO  → rewrite for the gaps, retrieve again, merge & dedupe, repeat.

Final synthesis delegates back to `engine.answer()` so CRAG / grounding /
reflection / save-back all still run — we just steer it to use OUR
accumulated retrieval set via a temporary `_retrieve_one` override.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..llm import OllamaClient, get_client
from ..query import QueryEngine, QueryResult
from .config import get_agentic_settings
from .feedback_rewriter import rewrite_for_gaps
from .planner import RetrievalPlan
from .search_fanout import execute_fanout
from .sufficient_context import (
    SufficientContextVerdict,
    evaluate_sufficient_context,
    quick_draft,
)

log = logging.getLogger(__name__)


def _to_snippet_dicts(pages: list[Any]) -> list[dict]:
    out: list[dict] = []
    for r in pages:
        meta = getattr(r, "meta", None) or {}
        out.append({
            "page_id": getattr(r, "page_id", ""),
            "title": meta.get("title") or getattr(r, "page_id", ""),
            "text": (getattr(r, "text", "") or "")[:2000],
        })
    return out


def _merge_dedupe(existing: list[Any], new: list[Any]) -> list[Any]:
    """Merge two lists of RetrievedPage objects, dedupe by page_id, keep best score."""
    by_id: dict[str, Any] = {}
    for r in existing + new:
        pid = getattr(r, "page_id", None)
        if not pid:
            continue
        prev = by_id.get(pid)
        if prev is None or getattr(r, "score", 0.0) > getattr(prev, "score", 0.0):
            by_id[pid] = r
    merged = sorted(by_id.values(), key=lambda r: getattr(r, "score", 0.0), reverse=True)
    return merged


async def _initial_retrieve(
    engine: QueryEngine,
    question: str,
    *,
    top_k: int,
    graph_expand: bool,
    plan: RetrievalPlan | None,
    hyde_text: str | None,
) -> tuple[list[Any], list[str]]:
    """Iteration-0 retrieval. Uses the planner's sub-tasks if multi-hop,
    otherwise a single-query call into the existing hybrid pipeline.

    Returns (retrieved_pages, queries_used).
    """
    if plan and plan.is_multi_hop and len(plan.sub_tasks) > 1:
        s = get_agentic_settings()
        # HyDE was generated against the ORIGINAL question; it's misleading as
        # a dense-embedding seed for sub-task queries that probe different
        # information. Sub-tasks fall back to their own raw text.
        results_by_idx = await execute_fanout(
            plan.sub_tasks,
            engine=engine,
            top_k=top_k,
            graph_expand=graph_expand,
            hyde_text=None,
            concurrency=s.agentic_fanout_concurrency,
        )
        merged: list[Any] = []
        for idx in sorted(results_by_idx):
            merged = _merge_dedupe(merged, results_by_idx[idx])
        queries_used = [t.query for t in plan.sub_tasks]
        return merged[: top_k * 2], queries_used

    batch = await engine._retrieve_one(
        question, top_k=top_k, graph_expand=graph_expand, hyde_text=hyde_text,
    )
    return list(batch or []), [question]


async def _final_synthesize(
    engine: QueryEngine,
    question: str,
    *,
    accumulated_pages: list[Any],
    answer_kwargs: dict,
) -> QueryResult:
    """Run the existing full synthesis pipeline on our accumulated pages.

    We temporarily override `engine._retrieve_one` so the existing
    `engine.answer()` body sees our accumulated pages instead of doing fresh
    retrieval. CRAG, synthesis, grounding, reflection, save-back all still run.

    Concurrency safety: monkey-patching a shared engine instance is racy when
    two requests hit the same `QueryEngine` at once (FastAPI/uvicorn workers
    do). We serialise the swap with a per-engine `asyncio.Lock`, lazily
    attached on first use. Synthesis is the dominant cost anyway, so the
    serialisation has negligible practical impact.
    """
    lock = getattr(engine, "_agentic_synth_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        engine._agentic_synth_lock = lock

    async def _stub(query_text: str, top_k: int, graph_expand: bool, hyde_text,
                    use_mmr: bool = True, **_kw):
        # Return the accumulated set; engine.answer() will pick its own top_k.
        # accumulated_pages is already score-sorted by the agentic loop.
        # Accept (and ignore) any extra retrieval kwargs the engine passes
        # (e.g. use_mmr) so the signature stays compatible with _retrieve_one.
        return list(accumulated_pages)

    async with lock:
        original_retrieve = engine._retrieve_one
        try:
            engine._retrieve_one = _stub  # type: ignore[method-assign]
            # Disable decompose / HyDE — we've already explored exhaustively
            # and don't want the engine doing fresh retrieval probes.
            kw = {
                "top_k": answer_kwargs.get("top_k", 5),
                "graph_expand": answer_kwargs.get("graph_expand", True),
                "use_hyde": False,
                "decompose": False,
                "save_back": answer_kwargs.get("save_back", True),
            }
            return await engine.answer(question, **kw)
        finally:
            engine._retrieve_one = original_retrieve  # type: ignore[method-assign]


async def agentic_answer(
    question: str,
    *,
    engine: QueryEngine,
    client: OllamaClient | None = None,
    plan: RetrievalPlan | None = None,
    max_iterations: int | None = None,
    coverage_threshold: float | None = None,
    top_k: int = 5,
    graph_expand: bool = True,
    save_back: bool = True,
) -> QueryResult:
    """Iterative SCA-driven retrieval loop.

    Args:
        question: user's question.
        engine: existing, unchanged `QueryEngine`.
        client: shared `OllamaClient` (defaults to global singleton).
        plan: optional `RetrievalPlan` from `planner.plan_retrieval`.
        max_iterations: overrides config.
        coverage_threshold: overrides config — SCA coverage_score above which
            we stop iterating.
        top_k, graph_expand, save_back: passed through to retrieval / synth.
    """
    s = get_agentic_settings()
    max_iter = max_iterations if max_iterations is not None else s.agentic_max_iterations
    cov_thr = coverage_threshold if coverage_threshold is not None else s.agentic_coverage_threshold
    c = client or engine.c or get_client()

    # HyDE for iteration 0 (mirror the existing pipeline default).
    try:
        hyde_text = await engine._hyde(question)
    except Exception:
        hyde_text = None

    accumulated, queries_used = await _initial_retrieve(
        engine, question,
        top_k=top_k, graph_expand=graph_expand,
        plan=plan, hyde_text=hyde_text,
    )

    verdicts: list[SufficientContextVerdict] = []
    sub_qs_hint = [t.query for t in plan.sub_tasks] if plan else [question]

    for it in range(max_iter):
        snippets = _to_snippet_dicts(accumulated)
        # Quick draft only useful from iteration 1+ once we have evidence.
        draft = ""
        if snippets:
            draft = await quick_draft(c, question=question, snippets=snippets,
                                      max_chars=s.agentic_draft_max_chars)

        verdict = await evaluate_sufficient_context(
            c,
            question=question,
            sub_questions=sub_qs_hint,
            retrieved_snippets=snippets,
            draft_answer=draft or None,
            iteration=it,
        )
        verdicts.append(verdict)

        log.info(
            "agentic iteration verdict",
            extra={"metadata": {
                "iteration": it,
                "is_sufficient": verdict.is_sufficient,
                "coverage": round(verdict.coverage_score, 3),
                "n_accumulated": len(accumulated),
                "missing": verdict.missing_aspects[:3],
            }},
        )

        if verdict.is_sufficient or verdict.coverage_score >= cov_thr:
            break
        if it == max_iter - 1:
            # Budget exhausted; synthesize from what we have.
            break

        # Gap-targeted re-search.
        new_queries = await rewrite_for_gaps(
            c,
            original_question=question,
            missing_aspects=verdict.missing_aspects,
            previous_queries=queries_used,
            suggested_queries=verdict.suggested_queries,
            max_queries=3,
        )
        if not new_queries:
            log.info("no new queries from rewriter; stopping early")
            break

        # Cap accumulated pages so context stays bounded across many iterations.
        # 4x top_k is enough for the synthesiser even after CRAG drops some;
        # excess pages are also dropped by the synth prompt budget anyway.
        max_accum = max(4 * top_k, 20)

        for q in new_queries:
            try:
                batch = await engine._retrieve_one(
                    q, top_k=s.agentic_per_iteration_top_k,
                    graph_expand=graph_expand, hyde_text=None,
                )
            except Exception as e:
                log.debug("agentic re-retrieve failed",
                          extra={"metadata": {"query": q[:80], "error": str(e)[:120]}})
                continue
            accumulated = _merge_dedupe(accumulated, list(batch or []))[:max_accum]
            queries_used.append(q)

    # Final synthesis through the existing pipeline.
    result = await _final_synthesize(
        engine, question,
        accumulated_pages=accumulated,
        answer_kwargs={"top_k": max(top_k, len(accumulated)),
                       "graph_expand": graph_expand, "save_back": save_back},
    )

    # Attach agentic telemetry into existing fields (non-breaking).
    if verdicts:
        last = verdicts[-1]
        agentic_summary = (
            f"agentic: {len(verdicts)} iter(s), final coverage="
            f"{last.coverage_score:.2f}, sufficient={last.is_sufficient}"
        )
        result.quality_issues = list(result.quality_issues or []) + [agentic_summary]
    if queries_used:
        merged_subs = list(dict.fromkeys((result.sub_queries or []) + queries_used))
        result.sub_queries = merged_subs[:12]

    return result
