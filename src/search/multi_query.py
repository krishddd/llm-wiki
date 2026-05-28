"""RAG-Fusion / multi-query paraphrasing.

Given the user's question, ask the LLM for 2-3 paraphrases that probe the same
intent from different angles. Retrieve for each paraphrase, then fuse the
ranked lists with Reciprocal-Rank-Fusion. Complements HyDE (which targets
semantic alignment) by widening intent coverage.

Cheap, single LLM call upstream of retrieval.
"""
from __future__ import annotations

import json
import logging
import re

from ..llm import OllamaClient

log = logging.getLogger(__name__)

_PARAPHRASE_SYSTEM = (
    "You write 2-3 paraphrases of a user's question. Each paraphrase probes the SAME intent "
    "from a different angle (different keywords, different phrasing, sometimes more specific or "
    "more general). Reply ONLY JSON: {\"queries\":[\"paraphrase 1\",\"paraphrase 2\",\"paraphrase 3\"]}"
)


def _extract(raw: str) -> list[str]:
    s = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    qs = [str(q).strip() for q in (d.get("queries") or []) if str(q).strip()]
    return qs[:3]


async def paraphrase(client: OllamaClient, question: str, *, max_paraphrases: int = 3) -> list[str]:
    """Return [original, paraphrase_1, paraphrase_2, ...]. On failure returns [original]."""
    if not question or len(question) < 10:
        return [question]
    try:
        raw = await client.qwen(question, system=_PARAPHRASE_SYSTEM, temperature=0.5)
        paras = _extract(raw)[:max_paraphrases]
    except Exception as e:
        log.debug("paraphrase failed", extra={"metadata": {"error": str(e)[:120]}})
        paras = []
    out = [question]
    for p in paras:
        if p.lower() != question.lower() and p not in out:
            out.append(p)
    return out


def rrf_fuse_pages(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """RRF-fuse multiple ranked lists of page_ids → single ranked list."""
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank_idx, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank_idx + 1)
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
