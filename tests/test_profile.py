"""Tests for the runtime-enforced profile/schema contract."""
from __future__ import annotations

import json

import pytest

from llm_wiki.wiki.pages import Page, write_page
from llm_wiki.wiki.profile import (
    DEFAULT_PROFILE,
    Profile,
    ProfileViolation,
    enforce,
    load_profile,
    validate_page,
)


def test_valid_current_schema_page_passes():
    fm = {"title": "RAG", "kind": "source", "type": "Source Document",
          "confidence": 0.9, "domain": "general", "tags": ["concept"], "entity_refs": ["RAG"]}
    r = validate_page(fm)
    assert r.ok and not r.errors


def test_unknown_kind_is_error():
    r = validate_page({"title": "X", "kind": "blogpost"})
    assert not r.ok
    assert any("blogpost" in e for e in r.errors)


def test_confidence_out_of_range_is_error():
    assert not validate_page({"title": "X", "kind": "source", "confidence": 1.5}).ok
    assert not validate_page({"title": "X", "kind": "source", "confidence": -0.1}).ok
    assert validate_page({"title": "X", "kind": "source", "confidence": 0.5}).ok


def test_list_fields_must_be_lists():
    r = validate_page({"title": "X", "kind": "source", "tags": "notalist"})
    assert not r.ok and any("tags" in e for e in r.errors)


def test_missing_required_field_is_error():
    assert not validate_page({"kind": "source"}).ok           # no title
    assert not validate_page({"title": "X"}).ok               # no kind


def test_unknown_domain_is_warning_not_error():
    r = validate_page({"title": "X", "kind": "source", "domain": "astrology"})
    assert r.ok                          # still valid
    assert any("astrology" in w for w in r.warnings)


def test_strict_mode_raises_warn_mode_does_not():
    bad = {"title": "X", "kind": "nope"}
    with pytest.raises(ProfileViolation):
        enforce("x.md", bad, mode="strict")
    assert enforce("x.md", bad, mode="warn").ok is False     # returns, no raise
    assert enforce("x.md", bad, mode="off").ok is True       # skipped entirely


def test_load_profile_partial_override(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"page_kinds": ["source", "memo"], "domains": ["general", "legal"]}),
                 encoding="utf-8")
    prof = load_profile(p)
    assert "memo" in prof.page_kinds and "legal" in prof.domains
    # untouched keys keep defaults
    assert prof.entity_types == DEFAULT_PROFILE.entity_types
    assert validate_page({"title": "M", "kind": "memo"}, profile=prof).ok


def test_load_profile_missing_or_broken_returns_default(tmp_path):
    assert load_profile(None) is DEFAULT_PROFILE
    assert load_profile(tmp_path / "nope.json") is DEFAULT_PROFILE
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_profile(bad) is DEFAULT_PROFILE


def test_write_page_warn_mode_still_writes(tmp_path, monkeypatch):
    # Default enforcement is "warn" → a schema-violating page still lands on disk.
    from llm_wiki import config as cfg
    from llm_wiki.wiki import pages as pages_mod
    s = cfg.Settings(profile_enforcement="warn")
    monkeypatch.setattr(pages_mod, "get_settings", lambda: s)
    path = tmp_path / "sources" / "bad.md"
    write_page(Page(path=path, frontmatter={"title": "B", "kind": "blogpost"}, body="hi"))
    assert path.exists()


def test_write_page_strict_mode_raises(tmp_path, monkeypatch):
    from llm_wiki import config as cfg
    from llm_wiki.wiki import pages as pages_mod
    s = cfg.Settings(profile_enforcement="strict")
    monkeypatch.setattr(pages_mod, "get_settings", lambda: s)
    path = tmp_path / "sources" / "bad.md"
    with pytest.raises(ProfileViolation):
        write_page(Page(path=path, frontmatter={"title": "B", "kind": "blogpost"}, body="hi"))
