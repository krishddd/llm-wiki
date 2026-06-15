"""Planner Agent — strategic retrieval planning.

Distinct from `QueryEngine._decompose()`: the planner emits sub-tasks with
explicit **dependencies** (multi-hop chains) and a chosen **strategy** per
sub-task. The fanout executor (`search_fanout.execute_fanout`) consumes the
plan and chains queries that depend on earlier results.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..llm import OllamaClient

log = logging.getLogger(__name__)


PLANNER_SYSTEM = (
    "You are a retrieval planner for a wiki Q&A system. Decompose the question "
    "into 1-4 sub-tasks. For each sub-task choose a strategy and declare any "
    "dependencies on earlier sub-tasks.\n\n"
    "Strategies:\n"
    "- keyword_search: BM25-friendly literal terms\n"
    "- entity_lookup: page lookup for a specific named entity\n"
    "- graph_traverse: requires following entity relations\n"
    "- fact_check: verify a specific factual claim\n\n"
    "Dependencies: if sub-task 2 needs an ID/name extracted from sub-task 1's "
    "results, set depends_on=[0]. The executor will inject results from listed "
    "indices into the dependent query as `{prev}`.\n\n"
    "Reply ONLY JSON:\n"
    '{"sub_tasks":[{"query":"...","strategy":"keyword_search","depends_on":[],'
    '"reason":"...","priority":1}, ...],'
    '"is_multi_hop":true|false,'
    '"estimated_iterations":1-3,'
    '"rationale":"..."}'
)


@dataclass
class SubTask:
    query: str
    strategy: str = "keyword_search"
    depends_on: list[int] = field(default_factory=list)
    reason: str = ""
    priority: int = 1


@dataclass
class RetrievalPlan:
    sub_tasks: list[SubTask] = field(default_factory=list)
    is_multi_hop: bool = False
    estimated_iterations: int = 1
    rationale: str = ""


def _extract_json(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?\n?", "", (s or "").strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_VALID_STRATEGIES = {"keyword_search", "entity_lookup", "graph_traverse", "fact_check"}


def _trivial_plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        sub_tasks=[SubTask(query=question, strategy="keyword_search", priority=1,
                           reason="atomic question")],
        is_multi_hop=False,
        estimated_iterations=1,
        rationale="trivial single-task plan",
    )


async def plan_retrieval(
    client: OllamaClient,
    question: str,
    *,
    intent: str = "synthesis",
    available_entities: list[str] | None = None,
) -> RetrievalPlan:
    """Produce a structured retrieval plan."""
    # Fast path: factual queries usually don't need planning.
    if intent == "factual":
        return _trivial_plan(question)

    ent_block = ""
    if available_entities:
        ents = ", ".join(available_entities[:20])
        ent_block = f"\n\nKNOWN ENTITIES (use these in queries when relevant): {ents}"

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"INTENT:\n{intent}{ent_block}\n\n"
        "Produce the plan JSON now."
    )

    try:
        raw = await client.qwen(prompt, system=PLANNER_SYSTEM, temperature=0.1)
    except Exception as e:
        log.warning("planner LLM call failed; falling back to trivial plan",
                    extra={"metadata": {"error": str(e)[:160]}})
        return _trivial_plan(question)

    data = _extract_json(raw) or {}
    raw_tasks = data.get("sub_tasks") or []
    sub_tasks: list[SubTask] = []
    for i, t in enumerate(raw_tasks[:4]):
        if not isinstance(t, dict):
            continue
        q = str(t.get("query", "")).strip()
        if not q:
            continue
        strat = str(t.get("strategy", "keyword_search")).strip()
        if strat not in _VALID_STRATEGIES:
            strat = "keyword_search"
        dep_raw = t.get("depends_on") or []
        deps: list[int] = []
        if isinstance(dep_raw, list):
            for d in dep_raw:
                try:
                    di = int(d)
                except (TypeError, ValueError):
                    continue
                if 0 <= di < i:  # can only depend on earlier indices
                    deps.append(di)
        try:
            prio = int(t.get("priority", i + 1))
        except (TypeError, ValueError):
            prio = i + 1
        sub_tasks.append(SubTask(
            query=q,
            strategy=strat,
            depends_on=deps,
            reason=str(t.get("reason", ""))[:200],
            priority=prio,
        ))

    if not sub_tasks:
        return _trivial_plan(question)

    is_multi_hop = bool(data.get("is_multi_hop")) or any(st.depends_on for st in sub_tasks)
    try:
        est_iter = max(1, min(3, int(data.get("estimated_iterations", 1))))
    except (TypeError, ValueError):
        est_iter = 2 if is_multi_hop else 1

    # NB: do NOT sort by priority. `depends_on` stores positional indices into
    # this list; reordering would silently misalign them. Priority is kept as a
    # hint on each SubTask; the fanout executor honours dependencies via wait
    # not via order.
    return RetrievalPlan(
        sub_tasks=sub_tasks,
        is_multi_hop=is_multi_hop,
        estimated_iterations=est_iter,
        rationale=str(data.get("rationale", ""))[:300],
    )
