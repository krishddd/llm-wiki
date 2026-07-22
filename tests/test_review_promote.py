"""Tests for the shared review→sources promotion helper.

Focus: the media re-embedding loop — on promotion the old `<review>#media#<n>` dense
vector is purged and the vector must be re-created under the new `<sources>#media#<n>`
id, else media retrieval points at a dense id that no longer exists.
"""
from __future__ import annotations

import pytest

from src.graph import KnowledgeGraph
from src.wiki.pages import read_page
from src.wiki.review_promote import promote_review_page


class FakeIndex:
    """Records upserts/deletes so a test can assert what got (re)indexed."""

    def __init__(self):
        self.docs: dict[str, str] = {}

    async def upsert(self, pid, text, meta=None):
        self.docs[pid] = text

    async def delete(self, pid):
        self.docs.pop(pid, None)


def _make_review_page(wiki_dir, name="foo"):
    (wiki_dir / "review").mkdir(parents=True, exist_ok=True)
    (wiki_dir / "sources").mkdir(parents=True, exist_ok=True)
    p = wiki_dir / "review" / f"{name}.md"
    p.write_text(
        "---\n"
        f"title: {name.title()}\n"
        "kind: source\n"
        "confidence: 0.5\n"
        "chunk_count: 1\n"
        "domain: general\n"
        "---\n"
        "Body about Docker containers.\n",
        encoding="utf-8",
    )
    return read_page(p)


@pytest.mark.asyncio
async def test_promote_reembeds_media_under_new_id(tmp_path):
    wiki = tmp_path / "wiki"
    page = _make_review_page(wiki)
    bm25, dense = FakeIndex(), FakeIndex()

    graph = KnowledgeGraph(tmp_path / "g.db")
    old_pid = "review/foo.md"
    # A media node + its old dense vector, as ingest would have created them.
    graph._conn.execute(
        "INSERT INTO media_nodes(page_id, kind, ordinal, content, caption, embedding_id) "
        "VALUES (?, 'table', 0, '| a | b |', 'demo table', ?)",
        (old_pid, f"{old_pid}#media#0"),
    )
    graph._conn.commit()
    dense.docs[f"{old_pid}#media#0"] = "old vector"

    new_pid = await promote_review_page(
        page, wiki_dir=wiki, bm25=bm25, dense=dense, graph=graph,
    )
    assert new_pid == "sources/foo.md"

    # Old media vector purged; new one re-embedded under the sources id.
    assert f"{old_pid}#media#0" not in dense.docs
    assert "sources/foo.md#media#0" in dense.docs
    assert "demo table" in dense.docs["sources/foo.md#media#0"]

    # Graph media row followed the page.
    row = graph._conn.execute(
        "SELECT page_id, embedding_id FROM media_nodes"
    ).fetchone()
    assert row == ("sources/foo.md", "sources/foo.md#media#0")
    graph.close()


@pytest.mark.asyncio
async def test_promote_no_media_is_noop(tmp_path):
    """Promotion without any media nodes must succeed and not fabricate media units."""
    wiki = tmp_path / "wiki"
    page = _make_review_page(wiki, name="bar")
    bm25, dense = FakeIndex(), FakeIndex()
    graph = KnowledgeGraph(tmp_path / "g.db")

    new_pid = await promote_review_page(
        page, wiki_dir=wiki, bm25=bm25, dense=dense, graph=graph,
    )
    assert new_pid == "sources/bar.md"
    assert "sources/bar.md#0" in dense.docs                      # chunk indexed
    assert not any("#media#" in k for k in dense.docs)           # no media fabricated
    graph.close()
