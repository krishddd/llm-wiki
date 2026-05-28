"""Reflection / critique pass over a draft answer.

After the synthesizer produces a first draft, we ask a fast model: 'is this
answer complete, well-cited, and free of hallucinations relative to the source
context?' If gaps are found, we surface them in `quality_issues` and optionally
trigger one re-synthesis with the gaps as guidance.

Cheap (one gemma call) and catches a class of failures that grounding alone
misses — e.g. answers that cite real pages but miss the question's actual
sub-parts, or under-detailed answers to multi-part questions.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..llm import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class Critique:
    is_complete: bool
    is_well_cited: bool
    quality_score: float           # 0-1
    issues: list[str]
    missing_aspects: list[str]     # things the answer should have addressed
    suggested_refinement: str      # short instruction for a re-synth pass


_CRITIQUE_SYSTEM = (
    "You critique a draft answer for completeness and citation quality, given the user's "
    "question and the source pages used. Be concrete and brief.\n"
    "Reply ONLY JSON:\n"
    '{"is_complete": true|false,'
    ' "is_well_cited": true|false,'
    ' "quality_score": 0.XX,'
    ' "issues": ["…"],'
    ' "missing_aspects": ["aspect 1", "aspect 2"],'
    ' "suggested_refinement": "one-sentence instruction for improving the answer"}'
)


def _extract(raw: str) -> dict:
    s = re.sub(r"^```(?:json)?\n?", "", (raw or "").strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


async def critique_answer(
    client: OllamaClient,
    *,
    question: str,
    answer: str,
    source_titles: list[str],
) -> Critique:
    if not answer.strip():
        return Critique(False, False, 0.0, ["empty answer"], [], "")
    titles = "\n".join(f"- {t}" for t in source_titles[:8])
    prompt = (
        f"USER QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER:\n{answer[:4000]}\n\n"
        f"SOURCE PAGES CITED:\n{titles}\n\n"
        "Critique the draft."
    )
    try:
        raw = await client.gemma(prompt, system=_CRITIQUE_SYSTEM, temperature=0.2)
        d = _extract(raw)
    except Exception as e:
        log.debug("critique failed", extra={"metadata": {"error": str(e)[:120]}})
        d = {}
    # `(d.get("quality_score") or 0.7)` would clobber a legitimate 0.0 → use a
    # sentinel pattern instead. Then clamp to [0, 1].
    raw_qs = d.get("quality_score")
    try:
        qs = float(raw_qs) if raw_qs is not None else 0.7
    except (TypeError, ValueError):
        qs = 0.7
    qs = max(0.0, min(1.0, qs))
    return Critique(
        is_complete=bool(d.get("is_complete", True)),
        is_well_cited=bool(d.get("is_well_cited", True)),
        quality_score=qs,
        issues=[str(x) for x in (d.get("issues") or [])][:5],
        missing_aspects=[str(x) for x in (d.get("missing_aspects") or [])][:5],
        suggested_refinement=str(d.get("suggested_refinement", ""))[:300],
    )


def should_refine(crit: Critique) -> bool:
    """Decide if a re-synthesis pass is worth doing."""
    if not crit.suggested_refinement:
        return False
    return (not crit.is_complete) or (crit.quality_score < 0.6) or len(crit.missing_aspects) >= 2
