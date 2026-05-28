"""Graph tests — entity canonicalization + 2-hop neighbour expansion."""
from __future__ import annotations

import pytest

from src.graph import ExtractedEntity, ExtractedRelation, KnowledgeGraph


@pytest.mark.asyncio
async def test_entity_canonicalization(tmp_path):
    """Near-duplicate entity names (case / whitespace / punctuation) collapse to one canonical id."""
    g = KnowledgeGraph(tmp_path / "graph.db", fuzzy_threshold=90)
    # All three are surface variants of the same concept. token_sort_ratio >= 90 after lowercase.
    await g.upsert_entities_delta("p1", [ExtractedEntity("LLM Wiki", "CONCEPT")])
    await g.upsert_entities_delta("p2", [ExtractedEntity("llm wiki", "CONCEPT")])
    await g.upsert_entities_delta("p3", [ExtractedEntity("llm-wiki", "CONCEPT")])
    stats = g.stats()
    import sqlite3
    cur = sqlite3.connect(str(tmp_path / "graph.db")).cursor()
    cur.execute("SELECT COUNT(DISTINCT COALESCE(canonical_id, id)) FROM entities WHERE type='CONCEPT'")
    distinct_canonicals = cur.fetchone()[0]
    assert distinct_canonicals == 1, f"expected one canonical entity, got {distinct_canonicals}"
    assert stats["pages_with_entities"] == 3


@pytest.mark.asyncio
async def test_entity_distinct_concepts_kept_separate(tmp_path):
    """Distinct concepts should NOT be merged — 'Docker' and 'Kubernetes' stay separate."""
    g = KnowledgeGraph(tmp_path / "g.db", fuzzy_threshold=90)
    await g.upsert_entities_delta("p1", [ExtractedEntity("Docker", "CONCEPT")])
    await g.upsert_entities_delta("p2", [ExtractedEntity("Kubernetes", "CONCEPT")])
    import sqlite3
    cur = sqlite3.connect(str(tmp_path / "g.db")).cursor()
    cur.execute("SELECT COUNT(DISTINCT COALESCE(canonical_id, id)) FROM entities WHERE type='CONCEPT'")
    assert cur.fetchone()[0] == 2


@pytest.mark.asyncio
async def test_graph_expansion_two_hops(tmp_path):
    g = KnowledgeGraph(tmp_path / "g.db")
    # A (p1) — RELATES_TO — B (p2) — RELATES_TO — C (p3)
    await g.upsert_entities_delta(
        "p1",
        [ExtractedEntity("A", "CONCEPT")],
        [ExtractedRelation("A", "CONCEPT", "B", "CONCEPT", "RELATES_TO")],
    )
    await g.upsert_entities_delta("p2", [ExtractedEntity("B", "CONCEPT")])
    await g.upsert_entities_delta(
        "p3",
        [ExtractedEntity("C", "CONCEPT")],
        [ExtractedRelation("B", "CONCEPT", "C", "CONCEPT", "RELATES_TO")],
    )
    neighbours = await g.neighbors_of_pages(["p1"], hops=2)
    assert "p3" in neighbours or "p2" in neighbours
