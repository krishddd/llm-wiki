"""Tests for the best-of-best RAG package: small-to-big chunks, LITM reorder,
claim verification, synthesis down-weighting, and RAPTOR-lite topic clustering."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki.query import _litm_reorder
from llm_wiki.search import hybrid as hybrid_module
from llm_wiki.search.chunks import (
    SUB_CHUNK_OVERLAP,
    SUB_CHUNK_TARGET,
    chunk_text,
    focused_text,
    indexable_text,
    matched_chunk_indices,
)
from llm_wiki.search.hybrid import hybrid_search
from llm_wiki.synth.claims import parse_claims
from llm_wiki.synth.verify import apply_verdicts, verify_claims
from llm_wiki.wiki.topics import build_topic_pages, greedy_cluster

# ───── chunks.py ─────


def test_matched_chunk_indices_parses_suffixes():
    hits = ["a.md#3", "a.md", "a.md#7", "b.md#media#2", "c.md#hq"]
    out = matched_chunk_indices(hits)
    assert out["a.md"] == [3, 7]
    assert out["b.md"] == []  # media hit votes for parent, no offset
    assert out["c.md"] == []


def test_focused_text_reaches_deep_content():
    """The matched chunk sits far beyond the old 4000-char horizon — focused_text must surface it."""
    filler = ("lorem ipsum dolor sit amet. " * 40 + "\n\n")
    body = filler * 10 + "THE-NEEDLE-FACT lives here.\n\n" + filler * 3
    title = "Long Doc"
    full = indexable_text(title, body, {})
    chunks = chunk_text(full, target_chars=SUB_CHUNK_TARGET, overlap=SUB_CHUNK_OVERLAP)
    needle_idx = next(i for i, c in enumerate(chunks) if "THE-NEEDLE-FACT" in c)
    assert needle_idx * (SUB_CHUNK_TARGET - SUB_CHUNK_OVERLAP) > 4000  # beyond old horizon

    out = focused_text(title, body, {}, [needle_idx], max_chars=4000)
    assert "THE-NEEDLE-FACT" in out
    assert len(out) <= 4000


def test_focused_text_invalid_idx_falls_back_empty():
    assert focused_text("T", "short body", {}, [99]) == ""
    assert focused_text("T", "short body", {}, []) == ""


# ───── hybrid small-to-big + synthesis down-weighting ─────


class StubChunkIndex:
    def __init__(self, ranked):
        self.ranked = ranked

    async def search(self, q, k=20):
        return self.ranked[:k]


class StubStore:
    def __init__(self, pages, metas=None):
        self.pages = pages
        self.metas = metas or {}

    async def get_text(self, pid):
        return self.pages.get(pid, "")

    async def get_meta(self, pid):
        return self.metas.get(pid, {"title": pid})


@pytest.mark.asyncio
async def test_hybrid_returns_matched_chunk_text(monkeypatch):
    def fake_rerank(query, candidates, k=5):
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(candidates[:k])]
    monkeypatch.setattr(hybrid_module, "rerank", fake_rerank)

    filler = "irrelevant padding sentence. " * 300          # ~8700 chars
    body = filler + "DEEP-ANSWER is documented here." + filler
    title = "big"
    full = indexable_text(title, body, {"title": title})
    chunks = chunk_text(full, target_chars=SUB_CHUNK_TARGET, overlap=SUB_CHUNK_OVERLAP)
    needle_idx = next(i for i, c in enumerate(chunks) if "DEEP-ANSWER" in c)

    bm25 = StubChunkIndex([f"big#{needle_idx}"])
    dense = StubChunkIndex([])
    ps = StubStore({"big": body}, metas={"big": {"title": title}})

    result = await hybrid_search(
        "q", bm25=bm25, dense=dense, page_store=ps,
        top_k_rerank=1, graph_expand=False, use_mmr=False,
    )
    assert result and "DEEP-ANSWER" in result[0].text


@pytest.mark.asyncio
async def test_hybrid_downweights_machine_pages(monkeypatch):
    def fake_rerank(query, candidates, k=5):
        return [(c, 0.9) for c in candidates[:k]]  # identical raw scores
    monkeypatch.setattr(hybrid_module, "rerank", fake_rerank)

    bm25 = StubChunkIndex(["synth", "primary"])
    dense = StubChunkIndex(["synth", "primary"])
    ps = StubStore(
        {"synth": "machine written body", "primary": "primary source body"},
        metas={"synth": {"title": "s", "kind": "synthesis"},
               "primary": {"title": "p", "kind": "source"}},
    )
    result = await hybrid_search(
        "q", bm25=bm25, dense=dense, page_store=ps,
        top_k_rerank=2, graph_expand=False, use_mmr=False, synth_downweight=0.85,
    )
    assert [r.page_id for r in result] == ["primary", "synth"]
    assert result[0].score > result[1].score


# ───── RRF k dial ─────


def test_rrf_fuse_k_is_a_precision_recall_dial():
    from llm_wiki.search.hybrid import _rrf_fuse

    # "spike" ranks #1 in one list, absent in the other; "consensus" ranks deep (#10)
    # in BOTH lists. Filler pads the lists so consensus sits at index 9.
    fill_a = [f"a{i}" for i in range(9)]
    fill_b = [f"b{i}" for i in range(9)]
    lists = [["spike"] + fill_a + ["consensus"], fill_b + ["consensus"]]

    low = dict(_rrf_fuse(lists, k=1))     # precision dial — steep rank penalty
    high = dict(_rrf_fuse(lists, k=60))   # recall/consensus dial — flat penalty

    # Low k rewards the single #1 spike over the doc both agree on mid-list.
    assert low["spike"] > low["consensus"]
    # High k lets cross-list consensus overtake the one-list spike.
    assert high["consensus"] > high["spike"]


@pytest.mark.asyncio
async def test_hybrid_threads_rrf_k(monkeypatch):
    seen = {}

    def spy_fuse(rank_lists, k=60):
        seen["k"] = k
        return [(rank_lists[0][0], 1.0)] if rank_lists and rank_lists[0] else []
    monkeypatch.setattr(hybrid_module, "_rrf_fuse", spy_fuse)
    monkeypatch.setattr(hybrid_module, "rerank", lambda q, c, k=5: [(x, 1.0) for x in c[:k]])

    bm25 = StubChunkIndex(["a"])
    dense = StubChunkIndex(["a"])
    ps = StubStore({"a": "body"}, metas={"a": {"title": "a", "kind": "source"}})
    await hybrid_search(
        "q", bm25=bm25, dense=dense, page_store=ps,
        top_k_rerank=1, graph_expand=False, use_mmr=False, rrf_k=11,
    )
    assert seen["k"] == 11


# ───── Matryoshka (MRL) truncation ─────


def test_truncate_mrl_shrinks_and_renormalizes():
    from llm_wiki.llm import truncate_mrl

    vec = [3.0, 4.0, 100.0, -50.0]      # leading dims carry the "important" semantics
    out = truncate_mrl(vec, 2)
    assert len(out) == 2
    # 3-4 right triangle → unit vector 0.6, 0.8 after L2 renormalization
    assert out == pytest.approx([0.6, 0.8])


def test_truncate_mrl_is_a_noop_when_disabled_or_shorter():
    from llm_wiki.llm import truncate_mrl

    assert truncate_mrl([1.0, 2.0, 3.0], 0) == [1.0, 2.0, 3.0]      # disabled
    assert truncate_mrl([1.0, 2.0], 5) == [1.0, 2.0]                 # never pads
    assert truncate_mrl([], 4) == []                                 # empty


# ───── strict grounding (XML isolation + negative constraint) ─────


def test_build_context_wraps_sources_in_xml_when_enabled():
    from llm_wiki.query import _build_context

    class _R:
        def __init__(self, pid, text, meta):
            self.page_id, self.text, self.meta = pid, text, meta

    retrieved = [
        _R("sources/a.md", "alpha body", {"title": 'A "quoted" page'}),
        _R("sources/b.md", "beta body", {"title": "B"}),
    ]
    ctx, cits = _build_context(retrieved, query="q", xml_grounding=True)
    assert '<source id="1" title="A \'quoted\' page">' in ctx  # quotes sanitized
    assert '<source id="2" title="B">' in ctx
    assert ctx.count("</source>") == 2
    assert "alpha body" in ctx and "beta body" in ctx

    # Off by default → no XML wrapper, existing behaviour preserved.
    plain, _ = _build_context(retrieved, query="q")
    assert "<source" not in plain


def test_grounding_fallback_is_wired_into_synth_prompt():
    from llm_wiki.query import GROUNDING_FALLBACK, SYNTH_SYSTEM

    assert GROUNDING_FALLBACK == "I cannot answer this based on the provided context."
    assert GROUNDING_FALLBACK in SYNTH_SYSTEM
    assert "<source" in SYNTH_SYSTEM


# ───── lost-in-the-middle reorder ─────


def test_litm_reorder_ends_loaded():
    assert _litm_reorder([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]
    assert _litm_reorder([1, 2]) == [1, 2]
    assert _litm_reorder([]) == []


# ───── claim verification ─────


class _Cit:
    def __init__(self, title, snippet):
        self.title = title
        self.snippet = snippet


class _JudgeClient:
    async def summarize(self, prompt, system=None, *, temperature=0.1):
        return '{"verdicts":[{"id":1,"verdict":"supported"},{"id":2,"verdict":"unsupported"}]}'


@pytest.mark.asyncio
async def test_verify_claims_downgrades_unsupported():
    answer = (
        "Docker is a container runtime [Docker Guide]^0.90. "
        "Docker was invented in 1802 [Docker Guide]^0.88."
    )
    claims = parse_claims(answer)
    assert len(claims) == 2
    cits = [_Cit("Docker Guide", "Docker is a container runtime released in 2013.")]

    verdicts = await verify_claims(_JudgeClient(), answer=answer, claims=claims, citations=cits)
    assert verdicts == ["supported", "unsupported"]

    display = apply_verdicts(claims, verdicts)
    assert claims[0].confidence == pytest.approx(0.90)
    assert claims[1].confidence == pytest.approx(0.88 * 0.35)
    assert display[1]["verdict"] == "unsupported"


# ───── RAPTOR-lite topics ─────


def test_greedy_cluster_separates_orthogonal_groups():
    a = [("a1", [1.0, 0.0]), ("a2", [0.98, 0.05]), ("a3", [0.99, 0.01]),
         ("b1", [0.0, 1.0]), ("b2", [0.02, 0.97]), ("b3", [0.01, 0.99])]
    clusters = greedy_cluster(a, sim_threshold=0.8)
    assert sorted(len(c) for c in clusters) == [3, 3]


class _TopicClient:
    async def embed(self, text):
        # deterministic 2-d embedding: docker docs → x-axis, biology docs → y-axis
        return [1.0, 0.0] if "docker" in text.lower() else [0.0, 1.0]

    async def reason(self, prompt, system=None, *, temperature=0.3):
        return (
            '{"title":"Container Infrastructure",'
            '"summary":"' + ("These pages cover container tooling and deployment. " * 8).strip() + '",'
            '"key_themes":["containers","deployment"]}'
        )


@pytest.mark.asyncio
async def test_build_topic_pages_writes_topic(tmp_path: Path):
    from llm_wiki.wiki.pages import Page, write_page
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    for i in range(3):
        write_page(Page(
            path=wiki / "sources" / f"docker-{i}.md",
            frontmatter={"title": f"Docker Doc {i}", "kind": "source",
                         "description": f"Docker container guide number {i}."},
            body=f"Docker content {i}. This body describes docker containers in detail.",
        ))
    write_page(Page(
        path=wiki / "sources" / "cells.md",
        frontmatter={"title": "Cell Biology", "kind": "source",
                     "description": "Biology of the cell."},
        body="Mitochondria are the powerhouse of the cell in this biology text.",
    ))

    result = await build_topic_pages(wiki_dir=wiki, client=_TopicClient(), min_cluster=3)
    assert result["topics_written"] == 1
    topic_files = list((wiki / "sources").glob("topic-*.md"))
    assert len(topic_files) == 1
    from llm_wiki.wiki.pages import read_page
    fm = read_page(topic_files[0]).frontmatter
    assert fm["kind"] == "topic"
    assert fm["type"] == "Topic Overview"
    assert fm["member_count"] == 3
    body = read_page(topic_files[0]).body
    assert "(/sources/docker-0.md)" in body

    # idempotent rebuild replaces rather than accumulates
    result2 = await build_topic_pages(wiki_dir=wiki, client=_TopicClient(), min_cluster=3)
    assert result2["removed_old"] == 1
    assert len(list((wiki / "sources").glob("topic-*.md"))) == 1
