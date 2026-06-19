"""Multimodal graph (Phase 1) tests — media_nodes + media↔entity edges.

Entity canonicalization needs `thefuzz`; the pure data/join logic is tested by
inserting entities/edges via direct SQL so it runs even where thefuzz is absent.
The canonicalization path is exercised separately under importorskip.
"""
from __future__ import annotations

import pytest

from src.graph import KnowledgeGraph


@pytest.fixture()
def graph(tmp_path):
    g = KnowledgeGraph(tmp_path / "graph.db")
    yield g
    g.close()


@pytest.mark.asyncio
async def test_add_and_get_media_nodes(graph) -> None:
    mid = await graph.add_media_node(
        page_id="p1.md", kind="table", content="| material | yield |\n|---|---|\n| 6061-T6 | 276 |",
        caption="Yield strengths", ordinal=0, embedding_id="p1.md#media#0",
    )
    assert mid > 0
    nodes = await graph.media_for_page("p1.md")
    assert len(nodes) == 1
    assert nodes[0]["kind"] == "table"
    assert nodes[0]["embedding_id"] == "p1.md#media#0"


@pytest.mark.asyncio
async def test_media_ordering_and_kinds(graph) -> None:
    await graph.add_media_node(page_id="p.md", kind="formula", content="$E=mc^2$", ordinal=2)
    await graph.add_media_node(page_id="p.md", kind="image", content="a diagram", ordinal=0)
    await graph.add_media_node(page_id="p.md", kind="code", content="def f(): ...", ordinal=1)
    nodes = await graph.media_for_page("p.md")
    assert [n["kind"] for n in nodes] == ["image", "code", "formula"]  # ordered by ordinal


@pytest.mark.asyncio
async def test_invalid_kind_rejected(graph) -> None:
    with pytest.raises(ValueError):
        await graph.add_media_node(page_id="p.md", kind="video", content="x")


@pytest.mark.asyncio
async def test_delete_media_for_page(graph) -> None:
    await graph.add_media_node(page_id="p.md", kind="table", content="t", ordinal=0)
    await graph.add_media_node(page_id="q.md", kind="code", content="c", ordinal=0)
    await graph.delete_media_for_page("p.md")
    assert await graph.media_for_page("p.md") == []
    assert len(await graph.media_for_page("q.md")) == 1


@pytest.mark.asyncio
async def test_media_for_entity_join_via_sql(graph) -> None:
    # Insert an entity + media node + edge directly (no thefuzz needed) and assert the
    # canonical-aware join surfaces the media node for that entity.
    mid = await graph.add_media_node(
        page_id="alloy.md", kind="table", content="6061-T6 properties", ordinal=0,
    )
    cur = graph._conn.cursor()
    cur.execute("INSERT INTO entities(name, type) VALUES (?, ?)", ("6061-T6", "CONCEPT"))
    eid = cur.lastrowid
    cur.execute(
        "INSERT INTO media_entities(media_id, entity_id, rel_type) VALUES (?, ?, ?)",
        (mid, eid, "MEASURES"),
    )
    graph._conn.commit()

    hits = await graph.media_for_entity("6061-T6")
    assert len(hits) == 1
    assert hits[0]["id"] == mid
    assert hits[0]["rel_type"] == "MEASURES"
    assert hits[0]["page_id"] == "alloy.md"
    # Unknown entity → no media.
    assert await graph.media_for_entity("nonexistent") == []


@pytest.mark.asyncio
async def test_link_media_entity_canonicalizes(graph) -> None:
    pytest.importorskip("thefuzz")
    mid = await graph.add_media_node(page_id="p.md", kind="image", content="stress diagram", ordinal=0)
    await graph.link_media_entity(media_id=mid, entity_name="Steel", entity_type="CONCEPT", rel_type="DEPICTS")
    hits = await graph.media_for_entity("Steel")
    assert len(hits) == 1
    assert hits[0]["rel_type"] == "DEPICTS"


@pytest.mark.asyncio
async def test_link_media_entity_rejects_unknown_entity_type(graph) -> None:
    pytest.importorskip("thefuzz")
    mid = await graph.add_media_node(page_id="p.md", kind="table", content="t", ordinal=0)
    # Unknown entity type is ignored (no edge created).
    await graph.link_media_entity(media_id=mid, entity_name="X", entity_type="BOGUS")
    assert await graph.media_for_entity("X") == []
