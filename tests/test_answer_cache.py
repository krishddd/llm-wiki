"""Tests for the semantic answer cache: cosine lookup, TTL, cap, confidence gate,
and the cited-page staleness guard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.query import QueryEngine
from src.wiki.answer_cache import SemanticAnswerCache, _cosine


class _FakePageStore:
    def __init__(self, existing):
        self.existing = set(existing)

    async def get_text(self, pid):
        return "body" if pid in self.existing else ""


def _engine(page_store, **flags):
    cfg = {"answer_cache_verify_pages": True, **flags}
    return QueryEngine(bm25=None, dense=None, page_store=page_store,
                       settings=SimpleNamespace(**cfg), client=SimpleNamespace())


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_lookup_hit_above_threshold(tmp_path):
    cache = SemanticAnswerCache(tmp_path / "c.db")
    cache.store("what is docker", [1.0, 0.0, 0.0], {"answer": "a container runtime"}, 0.9,
                max_entries=100, ttl_days=7)
    # Near-identical vector → hit.
    hit = cache.lookup([0.99, 0.01, 0.0], threshold=0.95, ttl_days=7)
    assert hit is not None and hit["answer"] == "a container runtime"
    assert hit["_cache_similarity"] >= 0.95
    cache.close()


def test_lookup_miss_below_threshold(tmp_path):
    cache = SemanticAnswerCache(tmp_path / "c.db")
    cache.store("q", [1.0, 0.0], {"answer": "x"}, 0.9, max_entries=100, ttl_days=7)
    assert cache.lookup([0.0, 1.0], threshold=0.95, ttl_days=7) is None
    cache.close()


def test_ttl_expiry_pruned_on_store(tmp_path):
    cache = SemanticAnswerCache(tmp_path / "c.db")
    # Backdate an entry well beyond the TTL.
    cache.store("old", [1.0, 0.0], {"answer": "stale"}, 0.9, max_entries=100, ttl_days=7)
    cache._conn.execute("UPDATE answer_cache SET created_at = created_at - ?", (30 * 86400,))
    cache._conn.commit()
    # A fresh store prunes expired rows; the stale one must be gone.
    cache.store("new", [0.0, 1.0], {"answer": "fresh"}, 0.9, max_entries=100, ttl_days=7)
    assert cache.lookup([1.0, 0.0], threshold=0.95, ttl_days=7) is None
    assert cache.lookup([0.0, 1.0], threshold=0.95, ttl_days=7)["answer"] == "fresh"
    cache.close()


def test_max_entries_cap(tmp_path):
    cache = SemanticAnswerCache(tmp_path / "c.db")
    for i in range(5):
        cache.store(f"q{i}", [float(i), 1.0], {"answer": str(i)}, 0.9, max_entries=3, ttl_days=7)
    count = cache._conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
    assert count == 3
    cache.close()


@pytest.mark.asyncio
async def test_freshness_guard_all_pages_present():
    eng = _engine(_FakePageStore(["sources/a.md", "sources/b.md"]))
    hit = {"retrieved_pages": ["sources/a.md", "sources/b.md"]}
    assert await eng._cache_hit_is_fresh(hit) is True


@pytest.mark.asyncio
async def test_freshness_guard_rejects_missing_cited_page():
    # One cited page was archived/rejected since caching → do NOT serve the hit.
    eng = _engine(_FakePageStore(["sources/a.md"]))
    hit = {"retrieved_pages": ["sources/a.md", "sources/gone.md"]}
    assert await eng._cache_hit_is_fresh(hit) is False


@pytest.mark.asyncio
async def test_freshness_guard_can_be_disabled():
    eng = _engine(_FakePageStore([]), answer_cache_verify_pages=False)
    hit = {"retrieved_pages": ["sources/gone.md"]}
    assert await eng._cache_hit_is_fresh(hit) is True


@pytest.mark.asyncio
async def test_freshness_guard_no_pages_is_fresh():
    eng = _engine(_FakePageStore([]))
    assert await eng._cache_hit_is_fresh({"retrieved_pages": []}) is True
