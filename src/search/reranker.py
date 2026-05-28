"""FlashRank cross-encoder reranker. Tiny (~30 MB), pure-Python, ms-latency."""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker  # lazy import — keeps test startup light
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="data/flashrank")
    return _ranker


@dataclass
class RerankCandidate:
    page_id: str
    text: str
    meta: dict | None = None


def rerank(query: str, candidates: list[RerankCandidate], k: int = 5) -> list[tuple[RerankCandidate, float]]:
    """Return top-k (candidate, score) sorted by relevance to query. No-op if candidates empty."""
    if not candidates:
        return []
    from flashrank import RerankRequest
    passages = [{"id": c.page_id, "text": c.text, "meta": c.meta or {}} for c in candidates]
    ranked = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
    by_id = {c.page_id: c for c in candidates}
    out: list[tuple[RerankCandidate, float]] = []
    for r in ranked[:k]:
        cand = by_id.get(r["id"])
        if cand is not None:
            out.append((cand, float(r.get("score", 0.0))))
    log.info("reranked", extra={"metadata": {"n_in": len(candidates), "n_out": len(out)}})
    return out
