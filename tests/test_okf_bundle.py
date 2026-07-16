"""OKF bundle loader / exporter / eval-harness tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval_harness import GoldenItem, eval_retrieval, load_golden, save_golden
from src.loaders.okf_loader import import_okf_bundle, validate_bundle
from src.wiki.okf_export import export_okf_bundle
from src.wiki.pages import Page, read_page, write_page


def _make_bundle(root: Path) -> Path:
    bundle = root / "acme-runbooks"
    bundle.mkdir(parents=True)
    (bundle / "index.md").write_text("# Index\n- [Deploy](deploy.md) - how to deploy\n", encoding="utf-8")
    (bundle / "deploy.md").write_text(
        "---\ntype: Playbook\ntitle: Deploy Runbook\ndescription: How to deploy the service.\n---\n\n"
        "# Deploy Runbook\n\nRun the pipeline, then verify [Rollback Plan](/rollback.md).\n"
        "See also [the docs](https://example.com/docs).\n",
        encoding="utf-8",
    )
    (bundle / "rollback.md").write_text(
        "---\ntype: Playbook\ntitle: Rollback Plan\n---\n\n# Rollback Plan\n\nRevert the release tag.\n",
        encoding="utf-8",
    )
    return bundle


def test_validate_bundle_flags_missing_type(tmp_path: Path):
    bundle = _make_bundle(tmp_path)
    (bundle / "broken.md").write_text("---\ntitle: No Type\n---\n\nbody\n", encoding="utf-8")
    report = validate_bundle(bundle)
    assert not report["conformant"]
    assert any("broken.md" in v for v in report["violations"])
    assert report["pages"] == 3  # index.md is reserved, not a concept page


@pytest.mark.asyncio
async def test_import_okf_bundle_preserves_and_stamps(tmp_path: Path):
    bundle = _make_bundle(tmp_path)
    wiki = tmp_path / "wiki"
    result = await import_okf_bundle(bundle, wiki_dir=wiki)
    assert result.pages_imported == 2
    imported = sorted((wiki / "sources").glob("okf-acme-runbooks-*.md"))
    assert len(imported) == 2
    fm = read_page(imported[0]).frontmatter
    assert fm["type"] == "Playbook"                      # producer key preserved
    assert fm["kind"] == "source"                        # unknown type → source
    assert fm["source"].startswith("okf-import:acme-runbooks/")
    assert fm["confidence"] == 0.85
    # external http link ignored; internal link counted
    assert result.links_found == 2


@pytest.mark.asyncio
async def test_export_okf_bundle_roundtrip(tmp_path: Path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "review").mkdir()
    write_page(Page(path=wiki / "sources" / "good.md",
                    frontmatter={"title": "Good", "kind": "source"},
                    body="A perfectly fine knowledge page about containers."))
    write_page(Page(path=wiki / "review" / "staged.md",
                    frontmatter={"title": "Staged", "kind": "source"},
                    body="Unaccepted draft content that must not ship."))
    (wiki / "log.md").write_text("# Operation Log\n", encoding="utf-8")

    out = tmp_path / "dist"
    stats = export_okf_bundle(wiki, out)
    assert stats["pages_exported"] == 1
    assert stats["conformant"]
    assert (out / "sources" / "good.md").exists()
    assert not (out / "review").exists()                 # staged work excluded
    assert 'okf_version: "0.1"' in (out / "index.md").read_text(encoding="utf-8")
    # the exported bundle re-imports cleanly
    report = validate_bundle(out)
    assert report["conformant"]


# ───── eval harness ─────


def test_golden_roundtrip(tmp_path: Path):
    items = [GoldenItem(question="What is X?", expected_pages=["sources/x.md"], keywords=["x"])]
    path = tmp_path / "golden.jsonl"
    save_golden(path, items)
    loaded = load_golden(path)
    assert loaded[0].question == "What is X?"
    assert loaded[0].expected_pages == ["sources/x.md"]


class _EvalIndex:
    def __init__(self, ranked):
        self.ranked = ranked

    async def search(self, q, k=20):
        return self.ranked[:k]


class _EvalStore:
    def __init__(self, pages):
        self.pages = pages

    async def get_text(self, pid):
        return self.pages.get(pid, "")

    async def get_meta(self, pid):
        return {"title": pid}


@pytest.mark.asyncio
async def test_eval_retrieval_metrics(tmp_path: Path):
    golden = [
        GoldenItem(question="find a", expected_pages=["a.md"]),
        GoldenItem(question="find z", expected_pages=["z.md"]),  # never retrieved
    ]
    bm25 = _EvalIndex(["a.md", "b.md"])
    dense = _EvalIndex(["b.md", "a.md"])
    store = _EvalStore({"a.md": "alpha content", "b.md": "beta content"})
    r = await eval_retrieval(golden, bm25=bm25, dense=dense, page_store=store,
                             top_k=2, use_mmr=False)
    assert r["items"] == 2
    assert r["recall_at_2"] == 0.5           # a.md found, z.md not
    assert r["hit_rate"] == 0.5
    assert 0 < r["mrr"] <= 1.0
    assert json.dumps(r)                      # serialisable for results file
