"""OKF v0.1 conformance tests — frontmatter stamping, reserved files, index/log format."""
from __future__ import annotations

import re
from pathlib import Path

from llm_wiki.wiki.index_md import rebuild_index
from llm_wiki.wiki.log_md import append_log
from llm_wiki.wiki.pages import (
    Page,
    PageStore,
    derive_description,
    okf_type_for,
    read_page,
    write_page,
)


def test_okf_type_mapping():
    assert okf_type_for("source") == "Source Document"
    assert okf_type_for("synthesis") == "Synthesis"
    assert okf_type_for("procedure") == "Procedure"
    assert okf_type_for("unknown-kind") == "Document"


def test_derive_description_skips_markdown_noise():
    body = (
        "# Heading\n\n"
        "| a | b |\n|---|---|\n"
        "The wiki stores curated knowledge as markdown pages. More text follows here.\n"
    )
    assert derive_description(body) == "The wiki stores curated knowledge as markdown pages."


def test_write_page_stamps_okf_fields(tmp_path: Path):
    p = tmp_path / "sources" / "sample.md"
    write_page(Page(path=p, frontmatter={"title": "Sample", "kind": "source"},
                    body="A short factual sentence about the sample document body."))
    page = read_page(p)
    fm = page.frontmatter
    assert fm["type"] == "Source Document"
    assert fm["description"].startswith("A short factual sentence")
    # ISO 8601 timestamp
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", fm["timestamp"])


def test_write_page_preserves_existing_type_and_description(tmp_path: Path):
    p = tmp_path / "x.md"
    write_page(Page(path=p, frontmatter={"kind": "source", "type": "Custom Type",
                                         "description": "Hand-written."},
                    body="Some body sentence that is long enough to be a description."))
    fm = read_page(p).frontmatter
    assert fm["type"] == "Custom Type"
    assert fm["description"] == "Hand-written."


def test_reserved_files_not_stamped_and_not_iterated(tmp_path: Path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "index.md").write_text("# Index\n", encoding="utf-8")
    write_page(Page(path=wiki / "sources" / "real.md",
                    frontmatter={"title": "Real", "kind": "source"},
                    body="Real page body sentence for the description field here."))
    ids = [pid for pid, _ in PageStore(wiki).iter_pages(("sources",))]
    assert ids == ["sources/real.md"]


def test_rebuild_index_okf_format(tmp_path: Path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    write_page(Page(path=wiki / "sources" / "a.md",
                    frontmatter={"title": "Page A", "kind": "source", "confidence": 0.9},
                    body="Page A explains the first concept in reasonable detail."))
    out = rebuild_index(wiki)
    text = out.read_text(encoding="utf-8")
    assert 'okf_version: "0.1"' in text
    assert "- [Page A](sources/a.md) - Page A explains the first concept in reasonable detail. (conf 0.9)" in text


def test_append_log_newest_first_date_groups(tmp_path: Path):
    append_log(tmp_path, "ingest", "Doc One", "chunks: 3")
    append_log(tmp_path, "query", "Question Two")
    text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert text.startswith("# Operation Log")
    # one date group containing both entries, newest first within the day
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})$", text, re.MULTILINE)
    assert len(dates) == 1
    assert text.index("**Query** Question Two") < text.index("**Ingest** Doc One — chunks: 3")


def test_stage_or_publish_slug_collision_disambiguates(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from llm_wiki.wiki.pages import stage_or_publish
    settings = SimpleNamespace(wiki_dir=tmp_path, confidence_threshold=0.6)

    fm_a = {"title": "Same Title", "kind": "source", "source": "raw/a.pdf", "confidence": 0.9}
    path_a, _ = stage_or_publish("Same Title", "body from document A goes here.", fm_a, settings=settings)
    # Different document, same title → must NOT overwrite A.
    fm_b = {"title": "Same Title", "kind": "source", "source": "raw/b.pdf", "confidence": 0.9}
    path_b, _ = stage_or_publish("Same Title", "body from document B goes here.", fm_b, settings=settings)
    assert path_a != path_b
    assert "body from document A" in path_a.read_text(encoding="utf-8")
    # Re-ingesting the SAME source overwrites its own page (no new file).
    fm_a2 = {"title": "Same Title", "kind": "source", "source": "raw/a.pdf", "confidence": 0.9}
    path_a2, _ = stage_or_publish("Same Title", "updated body from document A.", fm_a2, settings=settings)
    assert path_a2 == path_a


def test_append_log_preserves_legacy_content(tmp_path: Path):
    legacy = "# Operation Log\n\n## [2026-01-01 10:00] ingest | Old Entry\n\nold details\n"
    (tmp_path / "log.md").write_text(legacy, encoding="utf-8")
    append_log(tmp_path, "ingest", "New Entry")
    text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Old Entry" in text
    assert text.index("New Entry") < text.index("Old Entry")
