"""Question intent classifier — drives adaptive retrieval depth.

Four intent classes, each with a recommended retrieval profile:

- **factual**     — single-fact lookup. e.g. "What is the Free Energy Principle?"
                    → top_k=3, snippet mode, no graph expand needed
- **multi_hop**   — chain reasoning across 2-3 entities. e.g. "How does X relate to Y via Z?"
                    → top_k=6, graph_expand=True, paraphrasing helps
- **synthesis**   — compare/contrast/explain across multiple docs. e.g. "Compare A and B"
                    → top_k=8, full-page mode (not snippets), graph expand
- **exhaustive**  — "list all", "every", "all approaches". e.g. "List all evaluation tools"
                    → top_k=15, snippet mode, MMR diversification critical

Heuristic-first (zero LLM cost on obvious cases), LLM-fallback on ambiguous ones.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ..llm import OllamaClient

log = logging.getLogger(__name__)

Intent = Literal["factual", "multi_hop", "synthesis", "exhaustive"]


@dataclass
class IntentProfile:
    intent: Intent
    top_k: int
    full_page_mode: bool       # send full page text to synth, not 1500-char snippets
    graph_expand: bool
    use_mmr: bool
    rationale: str = ""


# Heuristic regexes — cheap signals before any LLM call.
_EXHAUSTIVE_RE = re.compile(
    r"\b(list (all|every|each)|enumerate|all (the )?(tools|approaches|methods|frameworks|techniques)|"
    r"every (single )?\w+|comprehensive list|exhaustive)\b",
    re.IGNORECASE,
)
_SYNTHESIS_RE = re.compile(
    r"\b(compare|contrast|differ(ence)?s? between|how do .* relate|trade-?offs?|"
    r"pros (and|&) cons|advantages and disadvantages|across (these )?(documents|papers|sources)|"
    r"summari[sz]e .* across|landscape of)\b",
    re.IGNORECASE,
)
_MULTIHOP_RE = re.compile(
    r"\b(via|through|because of|due to|leads? to|caus(es?|ed) by|connected (to|with)|"
    r"path (from|between)|chain of|two[- ]hop|multi[- ]?hop)\b",
    re.IGNORECASE,
)
_FACTUAL_RE = re.compile(
    r"^(what is|what are|who is|who are|when (is|was|did)|where (is|are|was)|"
    r"define|definition of|how many)\b",
    re.IGNORECASE,
)


def heuristic_intent(question: str) -> Intent | None:
    """Cheap rule-based classifier. Returns None if signals are mixed/absent."""
    q = question.strip()
    if _EXHAUSTIVE_RE.search(q):
        return "exhaustive"
    if _FACTUAL_RE.search(q) and len(q) < 90 and "and" not in q.lower():
        return "factual"
    if _SYNTHESIS_RE.search(q):
        return "synthesis"
    if _MULTIHOP_RE.search(q):
        return "multi_hop"
    return None


_INTENT_SYSTEM = (
    "Classify the user's question into ONE retrieval-intent class:\n"
    "- factual: single fact lookup, definition, who/what/when\n"
    "- multi_hop: requires linking 2-3 concepts across documents\n"
    "- synthesis: compare/contrast/summarize across multiple sources\n"
    "- exhaustive: must enumerate all items (list every X, all approaches, etc.)\n"
    'Reply ONLY JSON: {"intent":"factual|multi_hop|synthesis|exhaustive","reason":"…"}'
)


async def classify_intent_llm(client: OllamaClient, question: str) -> Intent:
    try:
        raw = await client.gemma(question, system=_INTENT_SYSTEM, temperature=0.1)
        s = re.sub(r"^```(?:json)?\n?", "", (raw or "").strip())
        s = re.sub(r"\n?```$", "", s)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            v = str(d.get("intent", "synthesis")).strip().lower()
            if v in ("factual", "multi_hop", "synthesis", "exhaustive"):
                return v  # type: ignore[return-value]
    except Exception as e:
        log.debug("LLM intent classification failed", extra={"metadata": {"error": str(e)[:120]}})
    return "synthesis"  # safest default — covers most questions adequately


_PROFILE_BY_INTENT: dict[Intent, IntentProfile] = {
    "factual":    IntentProfile(intent="factual",    top_k=3,  full_page_mode=False, graph_expand=False, use_mmr=False),
    "multi_hop":  IntentProfile(intent="multi_hop",  top_k=6,  full_page_mode=False, graph_expand=True,  use_mmr=True),
    "synthesis":  IntentProfile(intent="synthesis",  top_k=8,  full_page_mode=True,  graph_expand=True,  use_mmr=True),
    "exhaustive": IntentProfile(intent="exhaustive", top_k=15, full_page_mode=False, graph_expand=True,  use_mmr=True),
}


async def profile_for(client: OllamaClient, question: str, *, default_top_k: int = 5) -> IntentProfile:
    """Return an IntentProfile for retrieval. Tries heuristic first, then LLM fallback."""
    intent = heuristic_intent(question)
    rationale = "heuristic"
    if intent is None:
        intent = await classify_intent_llm(client, question)
        rationale = "llm"
    prof = _PROFILE_BY_INTENT[intent]
    return IntentProfile(
        intent=prof.intent,
        top_k=prof.top_k,
        full_page_mode=prof.full_page_mode,
        graph_expand=prof.graph_expand,
        use_mmr=prof.use_mmr,
        rationale=rationale,
    )
