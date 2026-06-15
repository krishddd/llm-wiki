"""Agentic RAG layer for LLM-Wiki.

This package wraps the existing `QueryEngine` / `hybrid_search` pipeline with an
iterative agentic loop driven by a Sufficient Context Agent (SCA), per Google
Research's "Sufficient Context" findings. No existing modules are modified.
"""
from .sufficient_context import SufficientContextVerdict, evaluate_sufficient_context
from .feedback_rewriter import rewrite_for_gaps
from .agentic_query import agentic_answer
from .planner import RetrievalPlan, SubTask, plan_retrieval
from .search_fanout import execute_fanout
from .orchestrator import QueryOrchestrator
from .config import AgenticSettings, get_agentic_settings

__all__ = [
    "SufficientContextVerdict",
    "evaluate_sufficient_context",
    "rewrite_for_gaps",
    "agentic_answer",
    "RetrievalPlan",
    "SubTask",
    "plan_retrieval",
    "execute_fanout",
    "QueryOrchestrator",
    "AgenticSettings",
    "get_agentic_settings",
]
