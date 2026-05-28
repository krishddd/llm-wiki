import pytest

from src.search.bm25_index import BM25Index


@pytest.mark.asyncio
async def test_bm25_upsert_search(tmp_path):
    idx = BM25Index(tmp_path / "bm25.pkl")
    await idx.upsert("p1", "docker fastapi integration")
    await idx.upsert("p2", "qwen reasoning model")
    await idx.upsert("p3", "docker alone")
    results = await idx.search("docker", k=5)
    assert results[0] in {"p1", "p3"}
    assert "p2" not in results


@pytest.mark.asyncio
async def test_bm25_persist(tmp_path):
    path = tmp_path / "bm25.pkl"
    idx = BM25Index(path)
    await idx.upsert("p1", "alpha beta")
    idx2 = BM25Index(path)
    results = await idx2.search("alpha", k=5)
    assert "p1" in results
