"""Top-level router. Decides: existing fast path vs. agentic loop.

Simple factual queries go straight to `QueryEngine.answer()` — zero overhead.
Multi-hop / synthesis / exhaustive go through the agentic loop with planning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import OllamaClient, get_client
from ..query import QueryEngine, QueryResult
from ..search.intent import classify_intent_llm, heuristic_intent
from .agentic_query import agentic_answer
from .config import get_agentic_settings
from .planner import plan_retrieval

log = logging.getLogger(__name__)


_COMPLEXITY_ORDER = {"factual": 0, "multi_hop": 1, "synthesis": 2, "exhaustive": 3}


@dataclass
class ComplexityAssessment:
    intent: str
    is_complex: bool
    suggested_iterations: int


def _suggested_iterations(intent: str) -> int:
    return {
        "factual": 1,
        "multi_hop": 3,
        "synthesis": 2,
        "exhaustive": 2,
    }.get(intent, 2)


class QueryOrchestrator:
    """Routes a question between the existing pipeline and the agentic loop.

    Usage:
        orch = QueryOrchestrator(engine)
        result = await orch.process("What are the specs of the server used in Project X?")
    """

    def __init__(self, engine: QueryEngine, *, client: OllamaClient | None = None):
        self.engine = engine
        self.c = client or engine.c or get_client()
        self.s = get_agentic_settings()

    async def _assess_complexity(self, question: str) -> ComplexityAssessment:
        intent = heuristic_intent(question)
        if intent is None:
            try:
                intent = await classify_intent_llm(self.c, question)
            except Exception as e:
                log.debug("intent LLM fallback failed",
                          extra={"metadata": {"error": str(e)[:120]}})
                intent = "synthesis"
        threshold = self.s.agentic_complexity_threshold
        is_complex = _COMPLEXITY_ORDER.get(intent, 2) >= _COMPLEXITY_ORDER.get(threshold, 1)
        return ComplexityAssessment(
            intent=intent,
            is_complex=is_complex,
            suggested_iterations=_suggested_iterations(intent),
        )

    async def process(
        self,
        question: str,
        *,
        top_k: int = 5,
        graph_expand: bool = True,
        save_back: bool = True,
        force_agentic: bool = False,
        force_simple: bool = False,
    ) -> QueryResult:
        if force_simple or not self.s.agentic_enabled:
            return await self.engine.answer(
                question, top_k=top_k, graph_expand=graph_expand, save_back=save_back,
            )

        assessment = await self._assess_complexity(question)
        log.info(
            "orchestrator complexity",
            extra={"metadata": {
                "intent": assessment.intent,
                "is_complex": assessment.is_complex,
                "force_agentic": force_agentic,
            }},
        )

        if not force_agentic and not assessment.is_complex:
            return await self.engine.answer(
                question, top_k=top_k, graph_expand=graph_expand, save_back=save_back,
            )

        # Build a plan. Cheap; falls back to a trivial one on failure.
        try:
            plan = await plan_retrieval(
                self.c, question,
                intent=assessment.intent,
                available_entities=None,
            )
        except Exception as e:
            log.debug("planner failed; continuing without plan",
                      extra={"metadata": {"error": str(e)[:120]}})
            plan = None

        return await agentic_answer(
            question,
            engine=self.engine,
            client=self.c,
            plan=plan,
            max_iterations=min(self.s.agentic_max_iterations, assessment.suggested_iterations),
            top_k=top_k,
            graph_expand=graph_expand,
            save_back=save_back,
        )
