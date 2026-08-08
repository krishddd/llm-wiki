"""Regression test for KnowledgeGraph.reassign_page_id (review→sources promotion).

When a staged page is promoted, its graph rows must follow it from the review id to
the sources id — otherwise pages_for_entity / contradiction detection silently skip
the page once the file has moved. Rows are inserted with raw SQL to avoid the
thefuzz-dependent canonicalization path.
"""
from __future__ import annotations

import pytest

from llm_wiki.graph import KnowledgeGraph


@pytest.mark.asyncio
async def test_reassign_page_id_repoints_all_tables(tmp_path):
    g = KnowledgeGraph(tmp_path / "g.db")
    c = g._conn.cursor()
    old, new = "review/foo.md", "sources/foo.md"
    c.execute(
        "INSERT INTO facts(subject_id,predicate,object_text,source_page,confidence) "
        "VALUES (1,'is','x',?,0.9)", (old,))
    c.execute("INSERT INTO page_entities(page_id,entity_id) VALUES (?,1)", (old,))
    c.execute(
        "INSERT INTO relations(src,dst,rel_type,source_page) VALUES (1,2,'RELATES_TO',?)", (old,))
    c.execute(
        "INSERT INTO media_nodes(page_id,kind,content,embedding_id) VALUES (?,'table','t',?)",
        (old, f"{old}#media#0"))
    c.execute("INSERT INTO page_access(page_id,access_count) VALUES (?,3)", (old,))
    g._conn.commit()

    await g.reassign_page_id(old, new)

    c = g._conn.cursor()
    assert c.execute("SELECT source_page FROM facts").fetchone()[0] == new
    assert c.execute("SELECT page_id FROM page_entities").fetchone()[0] == new
    assert c.execute("SELECT source_page FROM relations").fetchone()[0] == new
    page_id, emb = c.execute("SELECT page_id,embedding_id FROM media_nodes").fetchone()
    assert page_id == new and emb == f"{new}#media#0"
    pa = c.execute("SELECT page_id,access_count FROM page_access").fetchone()
    assert pa == (new, 3)
    g.close()


@pytest.mark.asyncio
async def test_reassign_page_id_noop_on_same_id(tmp_path):
    g = KnowledgeGraph(tmp_path / "g.db")
    c = g._conn.cursor()
    c.execute(
        "INSERT INTO facts(subject_id,predicate,object_text,source_page,confidence) "
        "VALUES (1,'is','x','sources/a.md',0.9)")
    g._conn.commit()
    await g.reassign_page_id("sources/a.md", "sources/a.md")  # no-op, must not raise
    assert c.execute("SELECT source_page FROM facts").fetchone()[0] == "sources/a.md"
    g.close()
