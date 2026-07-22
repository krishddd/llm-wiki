"""Tests for the feedback curator — classification, storage, promotion, preferences."""
from __future__ import annotations

import json

import pytest

from src.wiki.feedback import (
    FeedbackStore,
    _heuristic_kind,
    classify_feedback,
    promote_feedback,
    record_feedback,
)


class FakeClient:
    """Returns a queued classification for the reason role."""

    def __init__(self, kind="correction", actionable="RAG was introduced in 2020"):
        self.kind = kind
        self.actionable = actionable
        self.calls = 0

    async def reason(self, prompt, system=None, *, temperature=0.1):
        self.calls += 1
        return json.dumps({"kind": self.kind, "actionable": self.actionable})


def test_heuristic_drops_generic_acks():
    assert _heuristic_kind("thanks") == "noise"
    assert _heuristic_kind("ok") == "noise"
    assert _heuristic_kind("👍") == "noise"
    assert _heuristic_kind("") == "noise"
    # substantive feedback is not pre-filtered
    assert _heuristic_kind("Actually it was founded in 1998, not 2001") is None


@pytest.mark.asyncio
async def test_classify_noise_skips_llm():
    c = FakeClient()
    res = await classify_feedback(c, question="q", answer="a", text="thanks!")
    assert res["kind"] == "noise"
    assert c.calls == 0                      # heuristic short-circuit, no LLM call


@pytest.mark.asyncio
async def test_classify_correction_uses_llm():
    c = FakeClient(kind="correction", actionable="X shipped in 1998")
    res = await classify_feedback(c, question="q", answer="a", text="No, X shipped in 1998")
    assert res["kind"] == "correction" and "1998" in res["actionable"]
    assert c.calls == 1


@pytest.mark.asyncio
async def test_record_drops_noise(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    out = await record_feedback(store, FakeClient(), question="q", text="thanks")
    assert out["stored"] is False and out["kind"] == "noise"
    assert store.list() == []
    store.close()


@pytest.mark.asyncio
async def test_record_stores_candidate(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    out = await record_feedback(
        store, FakeClient(kind="preference", actionable="always be concise"),
        question="q", text="please be more concise",
    )
    assert out["stored"] and out["kind"] == "preference" and out["status"] == "candidate"
    cands = store.list(status="candidate")
    assert len(cands) == 1 and cands[0].actionable == "always be concise"
    store.close()


@pytest.mark.asyncio
async def test_promote_preference_becomes_active(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    fid = store.insert(question="q", answer_ref="", text="be concise",
                       kind="preference", actionable="always be concise")
    assert store.active_preferences() == []          # candidate, not yet active
    res = await promote_feedback(store, fid)
    assert res["ok"] and res["kind"] == "preference"
    assert store.active_preferences() == ["always be concise"]   # now injected
    store.close()


@pytest.mark.asyncio
async def test_promote_correction_writes_indexed_page(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    (tmp_path / "wiki" / "sources").mkdir(parents=True)

    class RecIndex:
        def __init__(self): self.ids = []
        async def upsert(self, pid, text, meta=None): self.ids.append(pid)
        async def delete(self, pid): pass

    bm25, dense = RecIndex(), RecIndex()
    fid = store.insert(question="When was RAG introduced?", answer_ref="", text="It was 2020",
                       kind="correction", actionable="RAG was introduced in 2020")
    res = await promote_feedback(store, fid, wiki_dir=tmp_path / "wiki", bm25=bm25, dense=dense)
    assert res["ok"] and "created sources/" in res["result"]
    # a curated page exists and was indexed as chunks
    pages = list((tmp_path / "wiki" / "sources").glob("*.md"))
    assert len(pages) == 1
    assert any(pid.startswith("sources/") and "#" in pid for pid in bm25.ids)
    # promoting again is a no-op
    res2 = await promote_feedback(store, fid, wiki_dir=tmp_path / "wiki", bm25=bm25, dense=dense)
    assert res2.get("note") == "already promoted"
    store.close()


@pytest.mark.asyncio
async def test_auto_promote_high_signal(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    out = await record_feedback(
        store, FakeClient(kind="preference", actionable="cite every claim"),
        question="q", text="you must cite every claim", auto_promote=True,
    )
    assert out["status"] == "promoted"
    assert store.active_preferences() == ["cite every claim"]
    store.close()
