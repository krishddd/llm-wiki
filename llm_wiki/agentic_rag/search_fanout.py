"""Parallel sub-task execution with dependency chaining.

Independent sub-tasks run concurrently via `asyncio.gather`. Sub-tasks that
declare `depends_on=[i, ...]` wait for those prerequisites, then have their
query rewritten to incorporate a hint extracted from the prerequisites'
top result titles.

The executor calls `QueryEngine._retrieve_one()` so HyDE / RRF / CRAG etc.
remain transparent — the agentic layer never bypasses existing retrieval.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .planner import SubTask

log = logging.getLogger(__name__)


def _extract_hint(retrieved: list[Any]) -> str:
    """Pull a short hint from prior results to chain into a dependent query."""
    if not retrieved:
        return ""
    parts: list[str] = []
    for r in retrieved[:2]:
        meta = getattr(r, "meta", None) or {}
        title = meta.get("title") or getattr(r, "page_id", "")
        if title:
            parts.append(str(title))
    return " ".join(parts).strip()


def _materialize_query(task: SubTask, dep_results: dict[int, list]) -> str:
    """Inject hints from dependent sub-tasks into `{prev}` placeholders."""
    q = task.query
    if not task.depends_on:
        return q
    hints: list[str] = []
    for idx in task.depends_on:
        hint = _extract_hint(dep_results.get(idx) or [])
        if hint:
            hints.append(hint)
    combined = " ".join(hints).strip()
    if "{prev}" in q:
        return q.replace("{prev}", combined)
    if combined:
        return f"{q} {combined}".strip()
    return q


async def execute_fanout(
    sub_tasks: list[SubTask],
    *,
    engine,                            # llm_wiki.query.QueryEngine — typed as Any to avoid cycle
    top_k: int = 5,
    graph_expand: bool = True,
    hyde_text: str | None = None,
    concurrency: int = 2,
) -> dict[int, list]:
    """Execute sub-tasks respecting `depends_on`. Returns {sub_task_index: results}.

    Sub-tasks without prerequisites are batched concurrently (capped by
    `concurrency`). Dependent sub-tasks run after their prerequisites finish.
    Each retrieval delegates to `engine._retrieve_one()` so the existing
    hybrid pipeline (BM25 + dense + RRF + rerank + MMR + graph) is reused.
    """
    if not sub_tasks:
        return {}

    sem = asyncio.Semaphore(max(1, concurrency))
    results: dict[int, list] = {}
    # One Event per sub-task index; set when that task's results are written.
    # Replaces a busy-poll on `results` and avoids deadlocks under low
    # concurrency because dependencies are awaited BEFORE acquiring `sem`,
    # so a dependent task never holds a slot while idle.
    done_events: dict[int, asyncio.Event] = {
        i: asyncio.Event() for i in range(len(sub_tasks))
    }

    async def _run_one(i: int, task: SubTask) -> None:
        # Wait for dependencies before acquiring the semaphore.
        for dep_i in task.depends_on:
            ev = done_events.get(dep_i)
            if ev is not None:
                await ev.wait()
        query = _materialize_query(task, results)
        async with sem:
            try:
                batch = await engine._retrieve_one(
                    query, top_k=top_k, graph_expand=graph_expand, hyde_text=hyde_text,
                )
                results[i] = list(batch or [])
                log.debug(
                    "fanout sub-task done",
                    extra={"metadata": {
                        "index": i, "strategy": task.strategy,
                        "n_results": len(results[i]),
                    }},
                )
            except Exception as e:
                log.warning(
                    "fanout sub-task failed",
                    extra={"metadata": {"index": i, "error": str(e)[:200]}},
                )
                results[i] = []
            finally:
                done_events[i].set()

    await asyncio.gather(*(_run_one(i, t) for i, t in enumerate(sub_tasks)))
    return results
