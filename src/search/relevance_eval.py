"""CRAG-style retrieval-relevance evaluator.

After hybrid_search returns, we ask a fast model whether each retrieved page is
actually relevant to the query (binary correct/incorrect/ambiguous). Then:

- If ≥1 page is "correct" → proceed with synthesis as normal
- If all are "incorrect" → return early with confidence=0 and a clear "no relevant
  pages found" answer (instead of having qwen hallucinate from junk context)
- If all "ambiguous" → proceed but downgrade max confidence ceiling

This catches the failure mode where retrieval surfaces topically-adjacent pages
that don't actually answer the question — synthesis on those produces confident-
sounding hallucinations that the self-grounding check can't catch (because the
model still cites the irrelevant pages).

Cheap: one ms-tier classifier prompt per page (gemma is fast enough).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ..llm import OllamaClient

log = logging.getLogger(__name__)

Verdict = Literal["correct", "incorrect", "ambiguous"]


@dataclass
class RelevanceVerdict:
    page_id: str
    verdict: Verdict
    score: float  # 0-1


_EVAL_SYSTEM = (
    "You judge whether a wiki page is relevant to a user question. "
    "Reply ONLY JSON: {\"relevance\":\"correct|ambiguous|incorrect\",\"score\":0.XX,\"reason\":\"…\"}\n"
    "- correct: page directly answers / contains evidence for the question\n"
    "- ambiguous: page touches related topics but isn't a direct answer\n"
    "- incorrect: page is off-topic; using it would mislead"
)


def _parse_verdict(raw: str) -> tuple[Verdict, float]:
    s = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return "ambiguous", 0.5
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "ambiguous", 0.5
    v = str(d.get("relevance", "ambiguous")).strip().lower()
    if v not in ("correct", "ambiguous", "incorrect"):
        v = "ambiguous"
    try:
        score = float(d.get("score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    return v, max(0.0, min(1.0, score))


async def evaluate_one(
    client: OllamaClient,
    question: str,
    page_id: str,
    page_text: str,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> RelevanceVerdict:
    sem = semaphore or asyncio.Semaphore(1)
    prompt = (
        f"QUESTION: {question}\n\n"
        f"PAGE [{page_id}] (truncated):\n{page_text[:2500]}\n\n"
        "Judge relevance."
    )
    try:
        async with sem:
            raw = await client.gemma(prompt, system=_EVAL_SYSTEM, temperature=0.1)
        verdict, score = _parse_verdict(raw)
    except Exception as e:
        log.debug("relevance eval failed", extra={"metadata": {"page": page_id, "error": str(e)[:120]}})
        verdict, score = "ambiguous", 0.5
    return RelevanceVerdict(page_id=page_id, verdict=verdict, score=score)


async def evaluate_batch(
    client: OllamaClient,
    question: str,
    pages: list[tuple[str, str]],  # [(page_id, text), ...]
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> list[RelevanceVerdict]:
    if not pages:
        return []
    sem = semaphore or asyncio.Semaphore(2)
    return await asyncio.gather(
        *[evaluate_one(client, question, pid, txt, semaphore=sem) for pid, txt in pages]
    )


@dataclass
class RetrievalDecision:
    """Outcome of CRAG-style filtering."""
    keep_page_ids: list[str]
    overall: Literal["correct", "ambiguous", "incorrect"]
    verdicts: list[RelevanceVerdict]
    confidence_ceiling: float  # max allowed final confidence given retrieval quality


def decide(verdicts: list[RelevanceVerdict], *, correct_threshold: float = 0.7, drop_threshold: float = 0.3) -> RetrievalDecision:
    """Filter pages and decide overall retrieval quality.

    - Drop pages with verdict==incorrect or score < drop_threshold
    - If any kept page is "correct" with score >= correct_threshold → "correct"
    - If all dropped → "incorrect"
    - Else → "ambiguous"
    """
    if not verdicts:
        return RetrievalDecision(keep_page_ids=[], overall="incorrect", verdicts=[], confidence_ceiling=0.0)

    keep = [v for v in verdicts if v.verdict != "incorrect" and v.score >= drop_threshold]
    if not keep:
        return RetrievalDecision(
            keep_page_ids=[],
            overall="incorrect",
            verdicts=verdicts,
            confidence_ceiling=0.2,
        )

    has_correct = any(v.verdict == "correct" and v.score >= correct_threshold for v in keep)
    overall: Literal["correct", "ambiguous", "incorrect"] = "correct" if has_correct else "ambiguous"
    ceiling = 1.0 if overall == "correct" else 0.7
    return RetrievalDecision(
        keep_page_ids=[v.page_id for v in keep],
        overall=overall,
        verdicts=verdicts,
        confidence_ceiling=ceiling,
    )
