"""Standalone synthesis-page writer.

Used by:
- `QueryEngine._save_synthesis_page` (save-back of high-confidence answers)
- `wiki.promote.promote_episodic_to_semantic` (Phase C tiered consolidation)
- `api.session_crystallize` (Phase F session digest)

Centralising this so all three call sites produce consistently formatted pages
that get indexed identically into BM25 + dense, and emit the same audit event.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..logging_config import audit
from .pages import Page, page_id_from_path, write_page

log = logging.getLogger(__name__)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s.strip().lower()).strip("-")
    return s[:80] or "page"


@dataclass
class SynthesisPageInputs:
    title: str
    body: str
    frontmatter: dict
    page_kind: str = "synthesis"  # "synthesis" | "promoted" | "crystallized" | "procedure"
    slug_prefix: str | None = None  # if None, derived from page_kind


async def write_synthesis_page(
    *,
    wiki_dir: Path,
    bm25,
    dense,
    inputs: SynthesisPageInputs,
) -> str | None:
    """Write a Markdown page to wiki/sources/, index it into BM25 + dense, audit-log.

    Returns the page-id (relative path from wiki_dir, posix-style) on success,
    or None on failure. All errors are caught and logged.
    """
    try:
        out_dir = Path(wiki_dir) / "sources"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = inputs.slug_prefix or inputs.page_kind
        path = out_dir / f"{prefix}-{_slug(inputs.title)}.md"

        # Defensive: ensure required frontmatter keys exist.
        fm = dict(inputs.frontmatter or {})
        fm.setdefault("title", inputs.title[:120])
        fm.setdefault("kind", inputs.page_kind)

        write_page(Page(path=path, frontmatter=fm, body=inputs.body))

        pid = page_id_from_path(path, Path(wiki_dir))

        # Index for retrieval.
        try:
            search_text = f"{fm['title']}\n{inputs.body}"
            if bm25 is not None:
                await bm25.upsert(pid, search_text)
            if dense is not None:
                await dense.upsert(
                    pid, search_text,
                    meta={"title": fm["title"], "kind": inputs.page_kind},
                )
        except Exception as e:
            log.warning("synth-page indexing failed",
                        extra={"metadata": {"error": str(e)[:200], "page_id": pid}})

        audit(
            log, "WIKI_WRITE", str(path),
            confidence=fm.get("confidence"),
            title=fm["title"],
            kind=inputs.page_kind,
        )
        return pid
    except Exception as e:
        log.warning("write_synthesis_page failed",
                    extra={"metadata": {"error": str(e)[:200], "title": inputs.title[:60]}})
        return None
