"""Import an external OKF bundle (Open Knowledge Format v0.1) as wiki pages.

OKF bundles are *curated* knowledge — deterministic answers someone already wrote
down — so unlike raw-document ingest they skip the summarise → score pipeline
entirely: concept pages are copied 1:1 (frontmatter preserved, as the spec
requires), stamped with import provenance, indexed at chunk level, and their
markdown links become RELATES_TO edges in the knowledge graph (the spec's
"consumers that build a graph view treat all links as directed edges").

No LLM calls — an import is fast and deterministic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..graph import ExtractedEntity, ExtractedRelation
from ..wiki.pages import RESERVED_FILENAMES, Page, read_page, write_page

log = logging.getLogger(__name__)

# OKF `type` → internal `kind`. Unknown types default to "source" (spec: consumers
# must handle unknown types gracefully).
KIND_BY_OKF_TYPE = {
    "source document": "source",
    "synthesis": "synthesis",
    "promoted synthesis": "promoted",
    "session digest": "crystallized",
    "procedure": "procedure",
    "topic overview": "topic",
}

# markdown links, excluding images: [label](target)
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)\s]+)\)")

# Curated-import default confidence: above the live threshold, below a verified 0.95.
IMPORT_CONFIDENCE = 0.85


@dataclass
class OKFImportResult:
    bundle: str
    pages_imported: int = 0
    pages_skipped: int = 0
    links_found: int = 0
    relations_added: int = 0
    errors: list[str] = field(default_factory=list)
    imported_page_ids: list[str] = field(default_factory=list)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s.strip().lower()).strip("-")
    return s[:80] or "page"


def validate_bundle(bundle_dir: Path) -> dict:
    """OKF conformance check: parseable frontmatter + non-empty `type` on every
    non-reserved .md file. Returns {conformant, pages, violations[]}."""
    bundle_dir = Path(bundle_dir)
    violations: list[str] = []
    pages = 0
    for p in sorted(bundle_dir.rglob("*.md")):
        if p.name.lower() in RESERVED_FILENAMES:
            continue
        pages += 1
        try:
            page = read_page(p)
        except Exception as e:
            violations.append(f"{p.relative_to(bundle_dir)}: unparseable ({str(e)[:80]})")
            continue
        if not page.frontmatter:
            violations.append(f"{p.relative_to(bundle_dir)}: missing frontmatter")
        elif not str(page.frontmatter.get("type", "")).strip():
            violations.append(f"{p.relative_to(bundle_dir)}: missing required `type`")
    return {"conformant": not violations, "pages": pages, "violations": violations}


async def import_okf_bundle(
    bundle_dir: Path,
    *,
    wiki_dir: Path,
    bm25=None,
    dense=None,
    graph=None,
    settings=None,
    client=None,
    confidence: float = IMPORT_CONFIDENCE,
) -> OKFImportResult:
    """Import every concept page of `bundle_dir` into wiki/sources/.

    Pages land as `okf-<bundle>-<slug>.md` with all producer frontmatter preserved
    (OKF requires consumers to keep unknown keys) plus import provenance. Body
    markdown links become CONCEPT→CONCEPT RELATES_TO edges when `graph` is given.
    """
    from ..config import get_settings

    bundle_dir = Path(bundle_dir)
    wiki_dir = Path(wiki_dir)
    bundle_name = _slug(bundle_dir.name)
    result = OKFImportResult(bundle=bundle_name)
    s = settings or get_settings()

    indexer = None
    if bm25 is not None or dense is not None:
        from ..ingest import Ingestor
        indexer = Ingestor(settings=s, client=client, graph=None, bm25=bm25, dense=dense)

    out_dir = wiki_dir / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(bundle_dir.rglob("*.md")):
        if p.name.lower() in RESERVED_FILENAMES:
            continue
        try:
            src_page = read_page(p)
        except Exception as e:
            result.errors.append(f"{p.name}: {str(e)[:120]}")
            continue
        fm = dict(src_page.frontmatter or {})
        if not fm:
            result.pages_skipped += 1
            continue

        rel = p.relative_to(bundle_dir).as_posix()
        title = str(fm.get("title") or p.stem.replace("-", " ").title())
        okf_type = str(fm.get("type") or "Document")
        fm.setdefault("title", title)
        fm["kind"] = KIND_BY_OKF_TYPE.get(okf_type.strip().lower(), "source")
        fm["source"] = f"okf-import:{bundle_name}/{rel}"
        fm["imported"] = date.today().isoformat()
        fm.setdefault("confidence", confidence)
        fm.setdefault("confidence_reason", "curated OKF bundle import (no LLM scoring)")

        out_path = out_dir / f"okf-{bundle_name}-{_slug(p.stem)}.md"
        write_page(Page(path=out_path, frontmatter=fm, body=src_page.body))
        pid = f"sources/{out_path.name}"
        result.pages_imported += 1
        result.imported_page_ids.append(pid)

        if indexer is not None:
            try:
                await indexer._index_page_chunks(pid, title, src_page.body, fm)
            except Exception as e:
                result.errors.append(f"index {pid}: {str(e)[:120]}")

        # Links → graph edges (concept-level RELATES_TO).
        links = _LINK_RE.findall(src_page.body)
        result.links_found += len(links)
        if graph is not None:
            if links:
                entities = [ExtractedEntity(name=title, type="CONCEPT")]
                relations = []
                for label, target in links[:30]:
                    if target.startswith(("http://", "https://", "mailto:")):
                        continue  # external citation, not a bundle edge
                    dst = label.strip()[:120]
                    if not dst or dst.lower() == title.lower():
                        continue
                    entities.append(ExtractedEntity(name=dst, type="CONCEPT"))
                    relations.append(ExtractedRelation(
                        src_name=title, src_type="CONCEPT",
                        dst_name=dst, dst_type="CONCEPT",
                        rel_type="RELATES_TO",
                    ))
                if relations:
                    try:
                        await graph.upsert_entities_delta(pid, entities, relations)
                        result.relations_added += len(relations)
                    except Exception as e:
                        result.errors.append(f"graph {pid}: {str(e)[:120]}")

    log.info(
        "OKF bundle imported",
        extra={"metadata": {
            "bundle": bundle_name, "imported": result.pages_imported,
            "relations": result.relations_added, "errors": len(result.errors),
        }},
    )
    return result
