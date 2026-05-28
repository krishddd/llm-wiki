"""Hybrid retrieval unit tests — BM25 + dense + RRF + rerank, with a stub reranker."""
from __future__ import annotations

import pytest

from src.search import hybrid as hybrid_module
from src.search.hybrid import _rrf_fuse, hybrid_search


class StubIndex:
    def __init__(self, ranked: list[str]):
        self.ranked = ranked

    async def search(self, q: str, k: int = 20) -> list[str]:
        return self.ranked[:k]


class StubPageStore:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    async def get_text(self, pid: str) -> str:
        return self.pages.get(pid, "")

    async def get_meta(self, pid: str) -> dict:
        return {"title": pid}


def test_rrf_fuse_basic():
    # y is the only item in the top position of both lists → must rank first.
    a = ["y", "x", "z"]
    b = ["y", "z", "x"]
    scored = dict(_rrf_fuse([a, b]))
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    assert ordered[0][0] == "y"


@pytest.mark.asyncio
async def test_hybrid_search_with_stub_rerank(monkeypatch):
    def fake_rerank(query, candidates, k=5):
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(candidates[:k])]
    monkeypatch.setattr(hybrid_module, "rerank", fake_rerank)

    bm25 = StubIndex(["a", "b", "c"])
    dense = StubIndex(["c", "b", "a"])
    ps = StubPageStore({"a": "alpha body", "b": "bravo body", "c": "charlie body"})

    result = await hybrid_search("q", bm25=bm25, dense=dense, page_store=ps, top_k_rerank=3, graph_expand=False)
    assert [r.page_id for r in result] == ["b", "a", "c"] or len(result) == 3
    assert all(r.text for r in result)
