"""Domain-routed dense index.

Wraps a primary (general) `DenseIndex` and an optional secondary (STEM) `DenseIndex`
that embeds with a stronger, notation-aware model (`model_embed_stem`). Vectors from
two different embedders live in incompatible spaces, so they MUST be kept in separate
collections — this wrapper owns that routing while presenting the same interface as a
plain `DenseIndex`, so `hybrid_search` and the API startup wiring barely change.

Routing:
- **upsert**: every page goes to the general index; quantitative pages (domain ∈
  {math, science, economics, engineering}) ALSO go to the STEM index.
- **search**: quantitative queries are served by the STEM index, everything else by
  the general index. Embedding-space consistency is guaranteed because each index
  embeds the query with its own model.

When `stem` is None (the default — `embed_stem_enabled=False`) this is a transparent
passthrough to the general index with zero behaviour change.
"""
from __future__ import annotations

import logging

from .domain import heuristic_domain

log = logging.getLogger(__name__)

# Domains that should use the STEM-specialized embedder.
_STEM_DOMAINS = {"math", "science", "economics", "engineering"}


def _is_stem_domain(domain: str | None) -> bool:
    return bool(domain) and domain in _STEM_DOMAINS


class DomainRoutedDenseIndex:
    def __init__(self, general, stem=None):
        self.general = general
        self.stem = stem

    # The general embed fn — used by HyDE/MMR helpers that read `_embed` directly.
    # Always general-space so cross-index comparisons stay consistent.
    @property
    def _embed(self):
        return self.general._embed

    @property
    def backend(self) -> str:
        b = self.general.backend
        return f"{b}+stem" if self.stem is not None else b

    async def upsert(self, page_id: str, text: str, meta: dict | None = None) -> None:
        await self.general.upsert(page_id, text, meta)
        if self.stem is not None and _is_stem_domain((meta or {}).get("domain")):
            try:
                await self.stem.upsert(page_id, text, meta)
            except Exception as e:
                log.warning("STEM dense upsert failed", extra={"metadata": {"error": str(e)[:160]}})

    async def delete(self, page_id: str) -> None:
        await self.general.delete(page_id)
        if self.stem is not None:
            try:
                await self.stem.delete(page_id)
            except Exception as e:
                log.debug("STEM dense delete failed", extra={"metadata": {"error": str(e)[:120]}})

    def _target_for_query(self, query: str):
        """Pick the index for a query by its detected domain."""
        if self.stem is not None and _is_stem_domain(heuristic_domain(query)):
            return self.stem, "stem"
        return self.general, "general"

    async def search(self, query: str, k: int = 20) -> list[str]:
        target, leg = self._target_for_query(query)
        if leg == "stem":
            log.debug("dense routed to STEM index", extra={"metadata": {"query": query[:80]}})
        return await target.search(query, k)

    async def route_search(self, query: str, hyde_text: str | None, k: int = 20) -> list[str]:
        """Domain-aware dense leg used by `hybrid_search`.

        For STEM queries: embed (HyDE text or raw query) with the STEM model and query
        the STEM index. For everything else: mirror the existing HyDE behaviour against
        the general index. Keeps each query in a single, consistent embedding space.
        """
        target, _leg = self._target_for_query(query)
        seed = hyde_text or query
        try:
            vec = await target._embed(seed)
            return await target.search_with_vec(vec, k=k)
        except Exception as e:
            log.warning("routed dense search failed, falling back to general",
                        extra={"metadata": {"error": str(e)[:160]}})
            return await self.general.search(query, k)

    async def search_with_vec(self, vec: list[float], k: int = 20) -> list[str]:
        # Precomputed vectors are general-space (callers use the general `_embed`).
        return await self.general.search_with_vec(vec, k)
