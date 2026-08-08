"""Regression test for gap #5: save-back must use the POST-verification confidence.

Save-back used to run *before* NLI-lite claim verification recalibrated the score, so
a synthesis persisted at the model's inflated self-score even when the verifier later
knocked it below the SAVE_BACK_CONF (0.80) gate. The fix reordered save-back to run
after verification.

This exercises the REAL retrieval stack (BM25 + numpy dense + FlashRank rerank + MMR)
end-to-end through `QueryEngine.answer()`, with a programmable client, and asserts:
  - all claims verified "supported"  → confidence stays high  → page IS saved
  - identical synthesis, claims "unsupported" → confidence drops → page is NOT saved

On the pre-fix ordering BOTH runs would have saved (at the 0.85 self-score), so the
"unsupported" case failing to save is exactly what proves the fix.
"""
from __future__ import annotations

import json

import pytest

from llm_wiki.query import QueryEngine
from llm_wiki.search.bm25_index import BM25Index
from llm_wiki.search.dense_index import DenseIndex
from llm_wiki.wiki.pages import Page, PageStore, write_page

SYNTH_JSON = json.dumps({
    "answer": ("Docker packages applications into containers [Seed Page A]^0.95. "
               "FastAPI serves the wiki API [Seed Page B]^0.95."),
    "summary": "Docker + FastAPI notes.",
    "key_points": ["Docker containerizes apps", "FastAPI serves the API"],
    "entities": ["Docker", "FastAPI"],
    "confidence": 0.85,
})


class ProgrammableClient:
    """Deterministic client: fixed synthesis; verify verdict is configurable."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    async def reason(self, prompt, system=None, *, temperature=0.3):
        if '"answer"' in (system or ""):          # SYNTH_SYSTEM
            return SYNTH_JSON
        return "{}"

    async def summarize(self, prompt, system=None, *, temperature=0.4):
        if "fact-checking judge" in (system or ""):   # verify_claims
            return json.dumps({"verdicts": [
                {"id": 1, "verdict": self.verdict},
                {"id": 2, "verdict": self.verdict},
            ]})
        return "[]"

    async def embed(self, text):
        # Bag-of-words vector over a tiny fixed vocab so the query is close to both
        # seed pages (deterministic retrieval, no model needed).
        vocab = ["docker", "fastapi", "container", "wiki", "api", "package"]
        low = text.lower()
        return [float(low.count(w)) + 0.1 for w in vocab]


def _seed_page(wiki_dir, name, title, body):
    p = wiki_dir / "sources" / f"{name}.md"
    write_page(Page(path=p, frontmatter={"title": title, "kind": "source", "confidence": 0.7}, body=body))
    return f"sources/{name}.md"


async def _build_engine(wiki_dir, data_dir, settings, client):
    (wiki_dir / "sources").mkdir(parents=True, exist_ok=True)
    bm25 = BM25Index(data_dir / "bm25.pkl")
    dense = DenseIndex(data_dir / "chroma", embed_fn=client.embed)
    pid_a = _seed_page(wiki_dir, "seed-a", "Seed Page A",
                       "Docker packages applications into containers for the wiki.")
    pid_b = _seed_page(wiki_dir, "seed-b", "Seed Page B",
                       "FastAPI serves the wiki API using Docker containers.")
    for pid, text in ((pid_a, "Docker container package wiki"),
                      (pid_b, "FastAPI wiki api Docker container")):
        await bm25.upsert(f"{pid}#0", text)
        await dense.upsert(f"{pid}#0", text, meta={"title": pid, "parent_id": pid})
    engine = QueryEngine(bm25=bm25, dense=dense, page_store=PageStore(wiki_dir), settings=settings, client=client)
    return engine


def _tune(settings):
    # Strip every optional LLM stage except per-claim confidence + claim verify, so the
    # only thing moving confidence is the verifier under test.
    settings.query_multi_query = False
    settings.query_relevance_eval = False
    settings.query_reflect = False
    settings.query_reflect_refine = False
    settings.route_solver_enabled = False
    settings.graph_multimodal_nodes = False
    settings.query_answer_cache = False
    settings.query_per_claim_confidence = True
    settings.query_claim_verify = True
    return settings


@pytest.mark.asyncio
async def test_saveback_persists_when_claims_supported(tmp_path, tmp_settings):
    s = _tune(tmp_settings)
    client = ProgrammableClient(verdict="supported")
    engine = await _build_engine(s.wiki_dir, s.data_dir, s, client)
    res = await engine.answer("What are Docker and FastAPI used for?",
                              adaptive=False, use_hyde=False, decompose=False)
    assert len(res.citations) >= 2
    assert res.confidence >= 0.80
    assert res.saved_page is not None                      # persisted


@pytest.mark.asyncio
async def test_saveback_skipped_when_claims_unsupported(tmp_path, tmp_settings):
    s = _tune(tmp_settings)
    client = ProgrammableClient(verdict="unsupported")
    engine = await _build_engine(s.wiki_dir, s.data_dir, s, client)
    res = await engine.answer("What are Docker and FastAPI used for?",
                              adaptive=False, use_hyde=False, decompose=False)
    # Verifier drags 0.85 self-score below the 0.80 gate → must NOT save-back.
    assert res.confidence < 0.80
    assert res.saved_page is None                          # the fix: no stale persist
