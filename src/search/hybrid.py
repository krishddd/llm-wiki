"""Hybrid retrieval pipeline: BM25 + dense (RRF fusion) → FlashRank rerank → graph expansion → MMR diversify → final top-k.

Advanced knobs:
- `hyde_text`: if supplied, its embedding is used for the dense leg while BM25 still uses the
  raw query (HyDE — Gao et al. 2022).
- `use_mmr`: if True, a final MMR pass (Carbonell & Goldstein 1998) diversifies the top-k.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .chunks import focused_text, matched_chunk_indices
from .mmr import MMRCandidate, mmr_select
from .reranker import RerankCandidate, rerank

log = logging.getLogger(__name__)


@dataclass
class RetrievedPage:
    page_id: str
    text: str
    score: float
    meta: dict


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-Rank-Fusion. Each ranked list contributes 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank_idx, page_id in enumerate(ranked):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


async def _dense_search(dense, query: str, hyde_text: str | None, k: int) -> list[str]:
    """Dense leg. Uses HyDE embedding if provided and supported, else raw query."""
    # Domain-routed index owns its own embedding-space decision (general vs. STEM).
    if hasattr(dense, "route_search"):
        return await dense.route_search(query, hyde_text, k=k)
    if hyde_text and hasattr(dense, "_embed") and hasattr(dense, "search_with_vec"):
        try:
            vec = await dense._embed(hyde_text)
            return await dense.search_with_vec(vec, k=k)
        except Exception as e:
            log.warning("HyDE dense path failed, falling back to raw query", extra={"metadata": {"error": str(e)}})
    return await dense.search(query, k=k)


async def hybrid_search(
    query: str,
    *,
    bm25,
    dense,
    page_store,
    graph=None,
    top_k_bm25: int = 20,
    top_k_dense: int = 20,
    top_k_rrf: int = 40,
    top_k_rerank: int = 5,
    graph_expand: bool = True,
    graph_hops: int = 2,
    graph_expand_cap: int = 10,
    hyde_text: str | None = None,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    use_chunk_context: bool = True,
    synth_downweight: float = 0.85,
) -> list[RetrievedPage]:
    """End-to-end retrieval.

    - `bm25` and `dense` must expose `async search(query, k) -> list[str]` (page_ids).
    - `page_store` exposes `get_text(page_id) -> str` and `get_meta(page_id) -> dict`.
    - `graph` (optional) exposes `neighbors_of_pages(page_ids, hops) -> list[str]`.
    - `hyde_text` (optional): hypothetical answer text; embedded and used for the dense leg.
    - `use_mmr`: final MMR diversification over the reranked+expanded set.
    - `use_chunk_context`: small-to-big — rerank/return the matched sub-chunks (plus
      neighbours) instead of the first 4000 chars of the parent page.
    - `synth_downweight`: score multiplier (<1) applied to machine-written pages
      (kind ∈ synthesis/promoted/crystallized) so save-back syntheses never outrank
      the primary sources they were derived from (anti-feedback-loop guardrail).
    """
    bm25_task = asyncio.create_task(bm25.search(query, k=top_k_bm25))
    dense_task = asyncio.create_task(_dense_search(dense, query, hyde_text, top_k_dense))
    bm25_ids, dense_ids = await asyncio.gather(bm25_task, dense_task)

    # Hierarchical chunk mapping: map chunk IDs back to parent page IDs, but KEEP
    # the matched sub-chunk offsets so we can hand the reranker/synthesiser the
    # text that actually matched (small-to-big retrieval) instead of page[:4000].
    parent_bm25 = []
    seen_bm25 = set()
    for cid in bm25_ids:
        pid = cid.split('#')[0]
        if pid not in seen_bm25:
            seen_bm25.add(pid)
            parent_bm25.append(pid)

    parent_dense = []
    seen_dense = set()
    for cid in dense_ids:
        pid = cid.split('#')[0]
        if pid not in seen_dense:
            seen_dense.add(pid)
            parent_dense.append(pid)

    chunk_hits = matched_chunk_indices(list(bm25_ids) + list(dense_ids)) if use_chunk_context else {}

    fused = _rrf_fuse([parent_bm25, parent_dense])[:top_k_rrf]
    if not fused:
        return []

    candidates = []
    for page_id, _ in fused:
        text = await page_store.get_text(page_id)
        if not text:
            continue
        cand_text = ""
        idxs = chunk_hits.get(page_id) or []
        if idxs:
            meta = await page_store.get_meta(page_id)
            title = str((meta or {}).get("title") or page_id)
            cand_text = focused_text(title, text, meta, idxs, max_chars=4000)
        candidates.append(RerankCandidate(page_id=page_id, text=cand_text or text[:4000]))

    # Rerank a slightly wider set when MMR is enabled, so MMR has room to diversify.
    rerank_k = max(top_k_rerank * 2, top_k_rerank + 5) if use_mmr else top_k_rerank
    top = rerank(query, candidates, k=rerank_k)

    if graph_expand and graph is not None and top:
        top_ids = [c.page_id for c, _ in top]
        extra_ids = await graph.neighbors_of_pages(top_ids, hops=graph_hops)
        extra_ids = [pid for pid in extra_ids if pid not in {c.page_id for c, _ in top}][:graph_expand_cap]
        if extra_ids:
            extra_cands = []
            for pid in extra_ids:
                text = await page_store.get_text(pid)
                if text:
                    extra_cands.append(RerankCandidate(page_id=pid, text=text[:4000]))
            combined = [c for c, _ in top] + extra_cands
            top = rerank(query, combined, k=rerank_k)

    # Anti-feedback-loop: down-weight machine-written pages so a save-back synthesis
    # never outranks the primary sources it was derived from.
    if synth_downweight < 1.0 and top:
        _MACHINE_KINDS = ("synthesis", "promoted", "crystallized")
        adjusted: list[tuple[RerankCandidate, float]] = []
        for c, s in top:
            meta = await page_store.get_meta(c.page_id)
            if str((meta or {}).get("kind", "")).lower() in _MACHINE_KINDS:
                s = s * synth_downweight
            adjusted.append((c, s))
        top = sorted(adjusted, key=lambda cs: cs[1], reverse=True)

    # Final step: MMR diversification down to top_k_rerank.
    if use_mmr and len(top) > top_k_rerank:
        mmr_cands = [MMRCandidate(page_id=c.page_id, text=c.text, relevance=float(s)) for c, s in top]
        embed_fn = getattr(dense, "_embed", None)
        selected = await mmr_select(mmr_cands, k=top_k_rerank, lambda_=mmr_lambda, embed_fn=embed_fn)
        selected_ids = {m.page_id for m in selected}
        # preserve rerank ordering among the selected subset
        top = [(c, s) for c, s in top if c.page_id in selected_ids][:top_k_rerank]
    else:
        top = top[:top_k_rerank]

    out: list[RetrievedPage] = []
    for cand, score in top:
        meta = await page_store.get_meta(cand.page_id)
        out.append(RetrievedPage(page_id=cand.page_id, text=cand.text, score=score, meta=meta or {}))

    # Phase B3: reinforce — bump access counters on the retrieved pages.
    # Best-effort; never blocks the response.
    if graph is not None and out:
        try:
            from ..config import get_settings
            from ..wiki.lifecycle import LifecycleConfig, mark_accessed
            s = get_settings()
            cfg = LifecycleConfig(
                half_life_days=getattr(s, "decay_half_life_days", 90.0),
                reinforcement_threshold=getattr(s, "reinforcement_threshold", 3),
                reinforcement_window_days=getattr(s, "reinforcement_window_days", 14),
                enabled=getattr(s, "lifecycle_enabled", True),
            )
            await mark_accessed(graph, [r.page_id for r in out], cfg=cfg)
        except Exception as e:
            log.debug("mark_accessed failed", extra={"metadata": {"error": str(e)[:120]}})

    log.info(
        "hybrid_search",
        extra={"metadata": {
            "query_len": len(query),
            "bm25": len(bm25_ids),
            "dense": len(dense_ids),
            "hyde": bool(hyde_text),
            "mmr": use_mmr,
            "final": len(out),
        }},
    )
    return out
