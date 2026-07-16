"""Smoke tests for the agentic RAG layer.

These tests stub the LLM client and the QueryEngine's retrieval / synthesis
internals so the iterative loop, SCA verdict, planner JSON parsing, and
feedback rewriter can be exercised without Ollama.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agentic_rag.agentic_query import agentic_answer
from src.agentic_rag.feedback_rewriter import rewrite_for_gaps
from src.agentic_rag.planner import plan_retrieval
from src.agentic_rag.search_fanout import SubTask, execute_fanout
from src.agentic_rag.sufficient_context import (
    SufficientContextVerdict,
    evaluate_sufficient_context,
)

# ───── Fakes ─────

@dataclass
class FakePage:
    page_id: str
    text: str
    score: float = 1.0
    meta: dict | None = None


class ProgrammableOllama:
    """Returns the next queued response per `client.reason` / `client.summarize` call."""

    def __init__(self, reason_responses=None, summarize_responses=None):
        self.reason_responses = list(reason_responses or [])
        self.summarize_responses = list(summarize_responses or [])
        self.reason_calls: list[str] = []
        self.summarize_calls: list[str] = []

    async def reason(self, prompt, system=None, *, temperature=0.3):
        self.reason_calls.append(prompt)
        if self.reason_responses:
            return self.reason_responses.pop(0)
        return '{"is_sufficient": true, "coverage_score": 1.0, "reason": "ok",' \
               ' "covered_aspects": [], "missing_aspects": [], "suggested_queries": []}'

    async def summarize(self, prompt, system=None, *, temperature=0.4):
        self.summarize_calls.append(prompt)
        if self.summarize_responses:
            return self.summarize_responses.pop(0)
        return "Quick draft answer."

    async def embed(self, text):
        return [0.0] * 8


# ───── SCA ─────

@pytest.mark.asyncio
async def test_sca_returns_insufficient_on_empty_snippets():
    client = ProgrammableOllama()
    v = await evaluate_sufficient_context(
        client, question="Q?", retrieved_snippets=[], iteration=0,
    )
    assert isinstance(v, SufficientContextVerdict)
    assert v.is_sufficient is False
    assert v.coverage_score == 0.0


@pytest.mark.asyncio
async def test_sca_parses_json_verdict():
    client = ProgrammableOllama(reason_responses=[
        '{"is_sufficient": false, "coverage_score": 0.55,'
        ' "reason": "missing dosing",'
        ' "covered_aspects": ["drug name"],'
        ' "missing_aspects": ["dosage", "frequency"],'
        ' "suggested_queries": ["aspirin dosage adult", "aspirin frequency"]}'
    ])
    v = await evaluate_sufficient_context(
        client,
        question="What is the dosage and frequency for aspirin?",
        retrieved_snippets=[{"page_id": "p1", "title": "Aspirin", "text": "..."}],
        iteration=0,
    )
    assert v.is_sufficient is False
    assert v.coverage_score == pytest.approx(0.55)
    assert "dosage" in v.missing_aspects
    assert len(v.suggested_queries) == 2


@pytest.mark.asyncio
async def test_sca_demotes_when_missing_but_high_score():
    client = ProgrammableOllama(reason_responses=[
        '{"is_sufficient": true, "coverage_score": 0.95,'
        ' "missing_aspects": ["dosage"], "covered_aspects": [],'
        ' "suggested_queries": []}'
    ])
    v = await evaluate_sufficient_context(
        client, question="Q?",
        retrieved_snippets=[{"page_id": "p1", "title": "T", "text": "x"}],
    )
    # Inconsistent verdict should be sanitised: not sufficient, score capped.
    assert v.is_sufficient is False
    assert v.coverage_score <= 0.7


# ───── Feedback rewriter ─────

@pytest.mark.asyncio
async def test_rewriter_uses_llm_and_dedupes():
    client = ProgrammableOllama(reason_responses=[
        '{"queries": ["aspirin dosage adult", "aspirin frequency daily"]}'
    ])
    out = await rewrite_for_gaps(
        client,
        original_question="What is the dosage and frequency for aspirin?",
        missing_aspects=["dosage", "frequency"],
        previous_queries=["aspirin"],
        suggested_queries=["aspirin dosage adult"],  # duplicate kept by LLM
    )
    assert "aspirin dosage adult" in out
    assert "aspirin frequency daily" in out
    # Previous query must not reappear
    assert all(q.lower() != "aspirin" for q in out)


@pytest.mark.asyncio
async def test_rewriter_falls_back_to_sca_suggestions_on_llm_failure():
    class BoomClient(ProgrammableOllama):
        async def reason(self, *a, **k):
            raise RuntimeError("ollama down")

    out = await rewrite_for_gaps(
        BoomClient(),
        original_question="Q?",
        missing_aspects=["x"],
        previous_queries=[],
        suggested_queries=["fallback query"],
    )
    assert out == ["fallback query"]


# ───── Planner ─────

@pytest.mark.asyncio
async def test_planner_factual_uses_trivial_plan():
    client = ProgrammableOllama()
    plan = await plan_retrieval(client, "What is Docker?", intent="factual")
    assert len(plan.sub_tasks) == 1
    assert plan.is_multi_hop is False
    # No LLM call required for factual fast path.
    assert client.reason_calls == []


@pytest.mark.asyncio
async def test_planner_parses_multi_hop_plan():
    client = ProgrammableOllama(reason_responses=[
        '{"sub_tasks": ['
        '{"query": "Project X server", "strategy": "keyword_search",'
        ' "depends_on": [], "reason": "find server id", "priority": 1},'
        '{"query": "{prev} specs hardware", "strategy": "entity_lookup",'
        ' "depends_on": [0], "reason": "lookup specs", "priority": 2}'
        '],'
        '"is_multi_hop": true, "estimated_iterations": 2, "rationale": "chain"}'
    ])
    plan = await plan_retrieval(client, "What are the specs of the server in Project X?",
                                intent="multi_hop")
    assert len(plan.sub_tasks) == 2
    assert plan.is_multi_hop is True
    assert plan.sub_tasks[1].depends_on == [0]


# ───── Search fanout ─────

class FakeEngine:
    """Minimal engine stub exposing _retrieve_one + _hyde."""

    def __init__(self, pages_by_query=None):
        self.pages_by_query = pages_by_query or {}
        self.calls: list[str] = []
        self.c = ProgrammableOllama()

    async def _retrieve_one(self, query_text, top_k, graph_expand, hyde_text):
        self.calls.append(query_text)
        return self.pages_by_query.get(query_text, [])

    async def _hyde(self, q):
        return None

    async def answer(self, question, **kw):
        # Pretend a real QueryResult — return a minimal stub object.
        from src.query import QueryResult
        return QueryResult(
            answer="final",
            answer_raw="final",
            summary="final summary",
            key_points=["p1"],
            citations=[],
            blocks=[],
            entities=[],
            confidence=0.9,
            retrieved_pages=[p.page_id for p in self.pages_by_query.get("__final__", [])],
            sub_queries=[question],
            grounded=True,
        )


@pytest.mark.asyncio
async def test_fanout_runs_independent_tasks_concurrently():
    engine = FakeEngine(pages_by_query={
        "Project X server": [FakePage("p1", "...", 1.0, {"title": "SRV-42"})],
        "Other thing": [FakePage("p2", "...", 0.9, {"title": "Other"})],
    })
    tasks = [
        SubTask(query="Project X server"),
        SubTask(query="Other thing"),
    ]
    res = await execute_fanout(tasks, engine=engine, top_k=3)
    assert len(res[0]) == 1 and res[0][0].page_id == "p1"
    assert len(res[1]) == 1 and res[1][0].page_id == "p2"


@pytest.mark.asyncio
async def test_fanout_chains_dependent_query():
    engine = FakeEngine(pages_by_query={
        "Project X server": [FakePage("p1", "...", 1.0, {"title": "SRV-42"})],
        "specs hardware SRV-42": [FakePage("p2", "...", 0.9, {"title": "Specs"})],
    })
    tasks = [
        SubTask(query="Project X server"),
        SubTask(query="specs hardware", depends_on=[0]),
    ]
    res = await execute_fanout(tasks, engine=engine, top_k=3)
    # Materialised query should append the prior title.
    assert any("SRV-42" in c for c in engine.calls)
    assert res[1][0].page_id == "p2"


# ───── Regression: bug-fix coverage ─────

@pytest.mark.asyncio
async def test_planner_preserves_depends_on_indices_under_unsorted_priorities():
    """Regression: a previous version sorted sub_tasks by priority, which
    silently misaligned `depends_on` indices that point into the original
    list position. After the fix, sub-tasks must appear in declared order
    even when priorities are out of order."""
    client = ProgrammableOllama(reason_responses=[
        '{"sub_tasks": ['
        '{"query": "first lookup", "strategy": "keyword_search",'
        ' "depends_on": [], "reason": "", "priority": 5},'
        '{"query": "second lookup", "strategy": "entity_lookup",'
        ' "depends_on": [0], "reason": "", "priority": 1}'
        '],'
        '"is_multi_hop": true, "estimated_iterations": 2, "rationale": ""}'
    ])
    plan = await plan_retrieval(client, "complex Q?", intent="multi_hop")
    # If we had sorted by priority, sub_tasks[0] would be 'second lookup' but
    # its depends_on=[0] would then point at itself — broken. Verify the
    # declared order is preserved.
    assert plan.sub_tasks[0].query == "first lookup"
    assert plan.sub_tasks[1].query == "second lookup"
    assert plan.sub_tasks[1].depends_on == [0]


@pytest.mark.asyncio
async def test_fanout_low_concurrency_no_deadlock():
    """Regression: with concurrency=1, the semaphore acquire must NOT happen
    before the dependency wait — otherwise A holds the only slot waiting on B
    and B can never acquire."""
    engine = FakeEngine(pages_by_query={
        "A": [FakePage("a", "...", 1.0, {"title": "A"})],
        "B A": [FakePage("b", "...", 0.9, {"title": "B"})],
    })
    tasks = [
        SubTask(query="A"),
        SubTask(query="B", depends_on=[0]),
    ]
    # If we hit the bug this awaits forever; pytest-asyncio default timeout
    # will surface it. We give it a generous wall-clock limit just in case.
    import asyncio as _asyncio
    res = await _asyncio.wait_for(
        execute_fanout(tasks, engine=engine, top_k=3, concurrency=1),
        timeout=5.0,
    )
    assert res[0][0].page_id == "a"
    assert res[1][0].page_id == "b"


# ───── End-to-end agentic loop ─────

@pytest.mark.asyncio
async def test_agentic_answer_one_iteration_sufficient():
    engine = FakeEngine(pages_by_query={
        "What is X?": [FakePage("p1", "X is foo.", 1.0, {"title": "X"})],
        "__final__": [FakePage("p1", "X is foo.", 1.0, {"title": "X"})],
    })
    # SCA returns sufficient on first iteration.
    engine.c = ProgrammableOllama(
        reason_responses=[
            '{"is_sufficient": true, "coverage_score": 0.95, "reason": "ok",'
            ' "covered_aspects": ["x"], "missing_aspects": [], "suggested_queries": []}'
        ],
        summarize_responses=["Draft: X is foo."],
    )

    result = await agentic_answer(
        "What is X?",
        engine=engine,
        client=engine.c,
        max_iterations=2,
        top_k=3,
        graph_expand=False,
        save_back=False,
    )
    assert result.answer == "final"
    # Should not have rewritten queries after a sufficient verdict.
    assert engine.calls == ["What is X?"]


@pytest.mark.asyncio
async def test_agentic_answer_iterates_when_insufficient():
    engine = FakeEngine(pages_by_query={
        "What are the server specs?": [
            FakePage("p1", "SRV-42 is in Project X.", 1.0, {"title": "Project X"})
        ],
        "SRV-42 specs hardware": [
            FakePage("p2", "SRV-42 has 64 cores.", 0.9, {"title": "SRV-42 Hardware"})
        ],
    })
    engine.c = ProgrammableOllama(
        reason_responses=[
            # iteration 0 — insufficient
            '{"is_sufficient": false, "coverage_score": 0.4, "reason": "missing specs",'
            ' "covered_aspects": ["server id"], "missing_aspects": ["hardware specs"],'
            ' "suggested_queries": ["SRV-42 specs hardware"]}',
            # rewriter
            '{"queries": ["SRV-42 specs hardware"]}',
            # iteration 1 — sufficient
            '{"is_sufficient": true, "coverage_score": 0.92, "reason": "got specs",'
            ' "covered_aspects": ["server id","specs"], "missing_aspects": [],'
            ' "suggested_queries": []}',
        ],
        summarize_responses=["draft 1", "draft 2"],
    )

    result = await agentic_answer(
        "What are the server specs?",
        engine=engine,
        client=engine.c,
        max_iterations=3,
        top_k=3,
        graph_expand=False,
        save_back=False,
    )
    # Should have retrieved twice: once for original, once for rewritten query.
    assert "What are the server specs?" in engine.calls
    assert "SRV-42 specs hardware" in engine.calls
    # Agentic telemetry surfaced via quality_issues.
    assert any("agentic" in s for s in (result.quality_issues or []))
