"""Review Autopilot tests — evidence-grounded accept / archive / annotate / skip."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.wiki.pages import Page, read_page, write_page
from src.wiki.review_autopilot import autopilot_review, entity_grounding


def _settings(**kw):
    base = dict(
        review_accept_threshold=0.70,
        review_reject_threshold=0.30,
        review_second_opinion=True,
        review_autopilot_max_pages=25,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class JudgeClient:
    """Programmable judge: summarize = primary, reason = second opinion."""

    def __init__(self, primary: dict, second: dict | None = None):
        self.primary = primary
        self.second = second or primary
        self.summarize_calls = 0
        self.reason_calls = 0

    async def summarize(self, prompt, system=None, *, temperature=0.1):
        self.summarize_calls += 1
        return json.dumps(self.primary)

    async def reason(self, prompt, system=None, *, temperature=0.1):
        self.reason_calls += 1
        return json.dumps(self.second)


class RecordingIndex:
    def __init__(self):
        self.upserts: list[str] = []

    async def upsert(self, pid, text, meta=None):
        self.upserts.append(pid)


def _make_wiki(tmp_path: Path, *, entity_refs=None, source_rel="raw/doc.md") -> Path:
    wiki = tmp_path / "wiki"
    (wiki / "review").mkdir(parents=True)
    (wiki / "sources").mkdir()
    raw = wiki / "raw"
    raw.mkdir()
    (wiki / "raw" / "doc.md").write_text(
        "Docker is a container runtime. FastAPI serves the wiki. "
        "Kubernetes orchestrates the containers across the cluster nodes.",
        encoding="utf-8",
    )
    write_page(Page(
        path=wiki / "review" / "staged-doc.md",
        frontmatter={
            "title": "Staged Doc", "kind": "source",
            "source": str(wiki / source_rel),
            "confidence": 0.4,
            "entity_refs": entity_refs if entity_refs is not None
            else ["Docker", "FastAPI", "Kubernetes"],
        },
        body="Summary: Docker runs containers, FastAPI serves the wiki, Kubernetes orchestrates.",
    ))
    return wiki


def test_entity_grounding_fraction():
    src = "Docker and FastAPI are mentioned here."
    assert entity_grounding(src, ["Docker", "FastAPI", "Kubernetes", "Zebra"]) == pytest.approx(0.5)
    # too few groundable entities → None (short names skipped)
    assert entity_grounding(src, ["ab", "cd"]) is None


@pytest.mark.asyncio
async def test_autopilot_accepts_high_composite(tmp_path: Path):
    wiki = _make_wiki(tmp_path)
    client = JudgeClient({"faithfulness": 0.95, "coverage": 0.9, "reasons": ["all claims present"]})
    bm25 = RecordingIndex()
    report = await autopilot_review(wiki_dir=wiki, client=client, bm25=bm25,
                                    settings=_settings())
    assert report["accepted"] == 1 and report["archived"] == 0
    assert not (wiki / "review" / "staged-doc.md").exists()
    dst = wiki / "sources" / "staged-doc.md"
    assert dst.exists()
    # Promoted page is indexed via small-to-big sub-chunks under its NEW sources id
    # (not a monolithic blob under the review id) — every unit is a `sources/...#<n>`.
    assert bm25.upserts, "expected the promoted page to be indexed"
    assert all(u.startswith("sources/staged-doc.md#") for u in bm25.upserts)
    assert not any(u.startswith("review/") for u in bm25.upserts)
    fm = read_page(dst).frontmatter
    assert fm["confidence"] >= 0.70            # confidence updated to composite
    assert fm["auto_review"]["verdict"] == "auto-accepted"   # provenance preserved on accept
    assert client.reason_calls == 0            # far from boundary → no second opinion


@pytest.mark.asyncio
async def test_autopilot_archives_low_composite(tmp_path: Path):
    wiki = _make_wiki(tmp_path, entity_refs=["Quantum", "Blockchain", "Astrology", "Nonsense"])
    client = JudgeClient({"faithfulness": 0.1, "coverage": 0.1, "reasons": ["claims not in source"]})
    report = await autopilot_review(wiki_dir=wiki, client=client, settings=_settings())
    assert report["archived"] == 1
    assert not (wiki / "review" / "staged-doc.md").exists()
    assert (wiki / "archive" / "staged-doc.md").exists()   # archived, never deleted


@pytest.mark.asyncio
async def test_autopilot_gray_zone_annotates_and_gets_second_opinion(tmp_path: Path):
    wiki = _make_wiki(tmp_path)
    # judge≈0.5 → composite in the gray zone; entities all ground (1.0) →
    # composite = 0.7*0.5 + 0.3*1.0 = 0.65 → within 0.08 of accept(0.70) → 2nd opinion
    client = JudgeClient(
        {"faithfulness": 0.5, "coverage": 0.5, "reasons": ["partial coverage"]},
        second={"faithfulness": 0.5, "coverage": 0.5, "reasons": ["agrees"]},
    )
    report = await autopilot_review(wiki_dir=wiki, client=client, settings=_settings())
    assert report["annotated"] == 1
    assert client.reason_calls == 1            # second opinion consulted
    staged = wiki / "review" / "staged-doc.md"
    assert staged.exists()                     # stays for the human
    ar = read_page(staged).frontmatter["auto_review"]
    assert ar["verdict"] == "needs-human"
    assert ar["composite"] == pytest.approx(0.65, abs=0.01)
    assert ar["reasons"]


@pytest.mark.asyncio
async def test_autopilot_skips_when_source_unreadable(tmp_path: Path):
    wiki = _make_wiki(tmp_path, source_rel="raw/does-not-exist.pdf")
    client = JudgeClient({"faithfulness": 0.99, "coverage": 0.99, "reasons": []})
    report = await autopilot_review(wiki_dir=wiki, client=client, settings=_settings())
    assert report["skipped"] == 1
    assert (wiki / "review" / "staged-doc.md").exists()    # untouched
    assert client.summarize_calls == 0                     # no evidence → no judging


@pytest.mark.asyncio
async def test_autopilot_only_page_filter(tmp_path: Path):
    wiki = _make_wiki(tmp_path)
    write_page(Page(
        path=wiki / "review" / "other.md",
        frontmatter={"title": "Other", "kind": "source",
                     "source": str(wiki / "raw" / "doc.md"), "confidence": 0.4,
                     "entity_refs": ["Docker", "FastAPI", "Kubernetes"]},
        body="Docker, FastAPI and Kubernetes summary.",
    ))
    client = JudgeClient({"faithfulness": 0.95, "coverage": 0.9, "reasons": []})
    report = await autopilot_review(wiki_dir=wiki, client=client, settings=_settings(),
                                    only_page="staged-doc.md")
    assert report["accepted"] == 1
    assert (wiki / "review" / "other.md").exists()         # untouched by the filter
