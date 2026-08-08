"""Ingest pipeline test with FakeOllama — verifies concurrency throttling + page write + graph update."""
from __future__ import annotations

import pytest

from llm_wiki.graph import KnowledgeGraph
from llm_wiki.ingest import Ingestor
from llm_wiki.search.bm25_index import BM25Index


@pytest.mark.asyncio
async def test_ingest_small_md(tmp_settings, fake_ollama, tmp_path):
    src = tmp_settings.raw_dir / "sample.md"
    src.write_text("Docker and FastAPI integrate well for the LLM wiki. " * 50, encoding="utf-8")

    bm25 = BM25Index(tmp_settings.data_dir / "bm25.pkl")
    graph = KnowledgeGraph(tmp_settings.data_dir / "graph.db")

    ingestor = Ingestor(settings=tmp_settings, client=fake_ollama, graph=graph, bm25=bm25, dense=None)
    result = await ingestor.ingest_file(src)
    assert result.is_live, f"expected live, got conf={result.confidence}"
    assert result.entities_added >= 1
    assert result.page_path
    assert (tmp_settings.wiki_dir / "sources").glob("*.md")
    # OKF conformance: source pages carry kind/type/description/resource/timestamp.
    from llm_wiki.wiki.pages import read_page
    fm = read_page(next((tmp_settings.wiki_dir / "sources").glob("*.md"))).frontmatter
    assert fm["kind"] == "source"
    assert fm["type"] == "Source Document"
    assert fm["resource"] == fm["source"]
    assert fm["timestamp"]
    assert fm["description"]
    # Doc2Query: hypothetical questions persisted + indexed as their own unit.
    assert fm["hypothetical_questions"]
    assert any(k.endswith("#hq") for k in bm25._docs)
    # throttling: FakeOllama tracks simultaneous calls; with semaphore=2 we must never exceed it.
    assert fake_ollama.max_concurrent_seen <= tmp_settings.max_concurrent_llm_req


@pytest.mark.asyncio
async def test_ingest_multi(tmp_settings, fake_ollama):
    for name in ("a.md", "b.md", "c.md"):
        (tmp_settings.raw_dir / name).write_text(f"Content for {name}. Docker FastAPI.", encoding="utf-8")
    ingestor = Ingestor(settings=tmp_settings, client=fake_ollama)
    results = await ingestor.ingest_many(list(tmp_settings.raw_dir.glob("*.md")))
    assert len(results) == 3
    assert all(r.is_live for r in results)
