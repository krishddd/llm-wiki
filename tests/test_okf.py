"""OKF v0.1 conformance tests — frontmatter stamping, reserved files, index/log format."""
from __future__ import annotations

import re
from pathlib import Path

from src.wiki.index_md import rebuild_index
from src.wiki.log_md import append_log
from src.wiki.pages import (
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


def test_append_log_preserves_legacy_content(tmp_path: Path):
    legacy = "# Operation Log\n\n## [2026-01-01 10:00] ingest | Old Entry\n\nold details\n"
    (tmp_path / "log.md").write_text(legacy, encoding="utf-8")
    append_log(tmp_path, "ingest", "New Entry")
    text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Old Entry" in text
    assert text.index("New Entry") < text.index("Old Entry")
