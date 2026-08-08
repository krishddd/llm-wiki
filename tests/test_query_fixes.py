"""Regression tests for the query-pipeline fixes.

Covers:
- `QueryEngine.procedures` defaults to None (was an AttributeError on engines built
  without the API's post-construction injection — e.g. the eval harness).
- Grounding check no longer treats loose substring overlap as a match (a short
  `[AI]` token must not ground against any title merely containing "ai").
"""
from __future__ import annotations

from llm_wiki.query import Citation, QueryEngine, _check_grounded, _token_matches_title


def test_query_engine_defaults_procedures_to_none():
    engine = QueryEngine(bm25=None, dense=None, page_store=None)
    assert engine.procedures is None


def test_token_matches_title_exact_and_word_boundary():
    titles = {"active inference for ai safety", "docker networking"}
    # Exact match.
    assert _token_matches_title("docker networking", titles)
    # Discriminating word-boundary substring (the token is a full multi-word phrase).
    assert _token_matches_title("active inference for ai safety", titles)


def test_token_matches_title_rejects_loose_substring():
    titles = {"active inference for ai safety"}
    # `ai` is a 2-char token — too short to ground against a longer title.
    assert not _token_matches_title("ai", titles)
    # A 4-char token that only appears mid-word must not match ("safe" in "safety").
    assert not _token_matches_title("safe", titles)


def test_check_grounded_flags_ungrounded_short_token():
    cits = [Citation(page="sources/x.md", title="Active Inference For AI Safety", snippet="")]
    # `[AI]` used to ground falsely against the long title; now it's ungrounded.
    grounded, n = _check_grounded("Claim about the field [AI].", cits)
    assert not grounded and n == 1


def test_check_grounded_accepts_exact_title_citation():
    cits = [Citation(page="sources/x.md", title="Active Inference For AI Safety", snippet="")]
    grounded, n = _check_grounded(
        "The framework is described in [Active Inference For AI Safety].", cits
    )
    assert grounded and n == 0
