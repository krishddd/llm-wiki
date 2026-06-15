"""Sufficient Context Agent (SCA).

The single highest-impact agentic component. Decides whether the retrieved
snippets are *complete enough* to fully answer the question — distinct from
CRAG (relevance per-page) and from `reflect.critique_answer` (post-hoc
critique on the final draft).

Returns a structured verdict with:
- a coverage score
- which sub-aspects ARE covered
- which sub-aspects are MISSING
- concrete suggested follow-up queries

The agentic loop in `agentic_query.py` consumes this verdict to decide
whether to re-retrieve.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..llm import OllamaClient

log = logging.getLogger(__name__)


SCA_SYSTEM = (
    "You are a Sufficient Context Agent for a research RAG system. Decide whether "
    "the supplied wiki snippets contain ENOUGH information to fully answer the "
    "user's question. You are NOT writing the final answer — you are auditing the "
    "evidence.\n\n"
    "Procedure:\n"
    "1. Break the question into its constituent sub-aspects (entities, attributes, "
    "comparisons, time ranges, causal links).\n"
    "2. For each sub-aspect, check whether at least one snippet contains evidence.\n"
    "3. List COVERED aspects and MISSING aspects.\n"
    "4. coverage_score = (#covered) / (#covered + #missing). 1.0 means complete.\n"
    "5. If anything is missing, propose 1–3 targeted follow-up queries that would "
    "retrieve the missing information. The queries must use concrete terms — "
    "entity names, IDs, attributes — not generic rephrasings.\n"
    "6. is_sufficient = (coverage_score >= 0.85) AND (no critical aspect missing).\n\n"
    "Reply ONLY with JSON matching this schema:\n"
    '{"is_sufficient": true|false,'
    ' "coverage_score": 0.0-1.0,'
    ' "reason": "<one sentence>",'
    ' "covered_aspects": ["..."],'
    ' "missing_aspects": ["..."],'
    ' "suggested_queries": ["..."]}\n'
    "No prose outside the JSON. No code fences."
)


@dataclass
class SufficientContextVerdict:
    is_sufficient: bool
    coverage_score: float
    reason: str
    missing_aspects: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    covered_aspects: list[str] = field(default_factory=list)
    iteration: int = 0


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


def _format_snippets(snippets: list[dict], max_chars_each: int = 700, max_snippets: int = 12) -> str:
    parts: list[str] = []
    for i, s in enumerate(snippets[:max_snippets], 1):
        title = s.get("title") or s.get("page_id") or f"snippet-{i}"
        text = (s.get("text") or "")[:max_chars_each]
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(parts) if parts else "(no snippets retrieved yet)"


def _trivial_sufficient(snippets: list[dict]) -> SufficientContextVerdict | None:
    """Fast path: no snippets at all → trivially insufficient."""
    if not snippets:
        return SufficientContextVerdict(
            is_sufficient=False,
            coverage_score=0.0,
            reason="No snippets retrieved.",
            missing_aspects=["all aspects — nothing was retrieved"],
            suggested_queries=[],
        )
    return None


async def evaluate_sufficient_context(
    client: OllamaClient,
    *,
    question: str,
    sub_questions: list[str] | None = None,
    retrieved_snippets: list[dict],
    draft_answer: str | None = None,
    iteration: int = 0,
) -> SufficientContextVerdict:
    """Decide whether retrieved context is sufficient.

    Args:
        client: shared OllamaClient.
        question: the original user question.
        sub_questions: decomposed sub-queries (hint for the SCA's aspect list).
        retrieved_snippets: list of `{page_id, title, text}` dicts.
        draft_answer: optional intermediate draft (helps the SCA spot under-supported
            sentences). Pass None on iteration 0.
        iteration: 0-based loop index, copied into the verdict for telemetry.
    """
    trivial = _trivial_sufficient(retrieved_snippets)
    if trivial is not None:
        trivial.iteration = iteration
        return trivial

    snippet_block = _format_snippets(retrieved_snippets)
    sub_q_block = ""
    if sub_questions:
        bullets = "\n".join(f"- {q}" for q in sub_questions[:6])
        sub_q_block = f"\n\nKNOWN SUB-ASPECTS (from decomposition):\n{bullets}"
    draft_block = ""
    if draft_answer:
        draft_block = f"\n\nDRAFT ANSWER (audit this for unsupported claims):\n{draft_answer[:1500]}"

    prompt = (
        f"QUESTION:\n{question}{sub_q_block}\n\n"
        f"RETRIEVED SNIPPETS:\n{snippet_block}{draft_block}\n\n"
        "Produce the JSON verdict now."
    )

    try:
        raw = await client.qwen(prompt, system=SCA_SYSTEM, temperature=0.1)
    except Exception as e:
        log.warning("SCA LLM call failed; defaulting to sufficient=False",
                    extra={"metadata": {"error": str(e)[:200]}})
        return SufficientContextVerdict(
            is_sufficient=False,
            coverage_score=0.5,
            reason=f"SCA call failed: {type(e).__name__}",
            iteration=iteration,
        )

    data = _extract_json(raw) or {}
    is_suff = bool(data.get("is_sufficient", False))
    try:
        score = float(data.get("coverage_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", ""))[:400]

    def _strlist(key: str, cap: int = 8) -> list[str]:
        v = data.get(key) or []
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()][:cap]

    missing = _strlist("missing_aspects")
    suggested = _strlist("suggested_queries", cap=5)
    covered = _strlist("covered_aspects")

    # Sanity: if the LLM says is_sufficient but listed missing aspects, trust missing.
    if missing and is_suff:
        is_suff = False
    # If the score is high but missing is non-empty, demote.
    if missing and score >= 0.9:
        score = min(score, 0.7)

    return SufficientContextVerdict(
        is_sufficient=is_suff,
        coverage_score=score,
        reason=reason,
        missing_aspects=missing,
        suggested_queries=suggested,
        covered_aspects=covered,
        iteration=iteration,
    )


QUICK_DRAFT_SYSTEM = (
    "You produce a SHORT draft answer (3-5 sentences, plain text) from wiki "
    "snippets. Do not invent facts. If a claim is not in the snippets, omit it. "
    "This draft will be audited for completeness, not shown to the user."
)


async def quick_draft(
    client: OllamaClient,
    *,
    question: str,
    snippets: list[dict],
    max_chars: int = 1200,
) -> str:
    """Lightweight intermediate draft used to feed the SCA's draft-review check."""
    if not snippets:
        return ""
    snippet_block = _format_snippets(snippets, max_chars_each=500, max_snippets=8)
    prompt = f"QUESTION:\n{question}\n\nSNIPPETS:\n{snippet_block}\n\nDraft now (3-5 sentences):"
    try:
        text = await client.gemma(prompt, system=QUICK_DRAFT_SYSTEM, temperature=0.2)
    except Exception as e:
        log.debug("quick_draft failed", extra={"metadata": {"error": str(e)[:120]}})
        return ""
    return (text or "").strip()[:max_chars]
