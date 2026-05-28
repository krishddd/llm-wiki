"""Maximum Marginal Relevance (MMR) — diversify a ranked candidate set.

Carbonell & Goldstein 1998. Balances relevance to the query with novelty vs already-selected items.

We use embedding cosine for similarity between candidates. Text-Jaccard is a dependency-free
fallback when no embedding is available.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class MMRCandidate:
    page_id: str
    text: str
    relevance: float  # score from the upstream reranker (higher == more relevant)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


async def mmr_select(
    candidates: list[MMRCandidate],
    *,
    k: int,
    lambda_: float = 0.7,
    embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
) -> list[MMRCandidate]:
    """Select k items maximising λ·relevance − (1−λ)·max_sim_to_selected.

    λ=1.0 → pure relevance (like argmax). λ=0.0 → pure novelty.
    0.7 is a sensible default for wiki synthesis.
    """
    if not candidates:
        return []
    if k >= len(candidates):
        return candidates

    # Pre-compute candidate representations (embeddings if possible, else token sets).
    emb_cache: dict[str, list[float]] = {}
    tok_cache: dict[str, set[str]] = {}
    for c in candidates:
        if embed_fn is not None:
            try:
                emb_cache[c.page_id] = await embed_fn(c.text[:4000])
            except Exception:
                tok_cache[c.page_id] = _tokenize(c.text[:4000])
        else:
            tok_cache[c.page_id] = _tokenize(c.text[:4000])

    def sim(a: MMRCandidate, b: MMRCandidate) -> float:
        if a.page_id in emb_cache and b.page_id in emb_cache:
            return _cos(emb_cache[a.page_id], emb_cache[b.page_id])
        return _jaccard(
            tok_cache.get(a.page_id) or _tokenize(a.text[:4000]),
            tok_cache.get(b.page_id) or _tokenize(b.text[:4000]),
        )

    # Normalise relevance to [0,1] so the λ trade-off is scale-free.
    rels = [c.relevance for c in candidates]
    lo, hi = min(rels), max(rels)
    span = (hi - lo) or 1.0
    norm = {c.page_id: (c.relevance - lo) / span for c in candidates}

    remaining = candidates[:]
    selected: list[MMRCandidate] = []
    # Seed with the single most relevant.
    remaining.sort(key=lambda c: c.relevance, reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_score = -1e9
        best_idx = 0
        for i, c in enumerate(remaining):
            max_sim = max(sim(c, s) for s in selected)
            score = lambda_ * norm[c.page_id] - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected
