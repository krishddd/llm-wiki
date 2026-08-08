"""Shared review→sources promotion.

Both accept paths — the Review Autopilot (`review_autopilot._accept`) and the human
`POST /review/{id}/accept` endpoint — funnel through `promote_review_page` so a
promoted page becomes a first-class retrieval citizen instead of a monolithic blob:

- the stale `review/*` retrieval units (parent, `#hq`, `#<n>`, `#media#<n>`) are
  deleted, using the frontmatter-stamped `chunk_count` for an exact sweep;
- the page is re-indexed under its NEW `sources/*` id via the same small-to-big
  chunker used at ingest (contextual preamble, dense metadata, Doc2Query `#hq`);
- the knowledge-graph rows (facts / entities / relations / media) are re-pointed
  from the old id to the new one, so `pages_for_entity` and the contradiction
  detector keep seeing the page after the file has moved.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from .index_md import rebuild_index
from .pages import Page, page_id_from_path, write_page
from .reindex import delete_page_index, index_doc2query, index_page_chunks

log = logging.getLogger(__name__)


async def promote_review_page(
    page: Page,
    *,
    wiki_dir: Path,
    bm25=None,
    dense=None,
    graph=None,
    new_confidence: float | None = None,
    extra_frontmatter: dict | None = None,
) -> str:
    """Move a staged review page to `sources/`, re-index it, and re-point graph rows.

    `page` is the in-memory Page (already read from `wiki/review/`). Returns the new
    page id (`sources/<name>.md`). Callers are responsible for audit logging.
    """
    wiki_dir = Path(wiki_dir)
    old_path = page.path
    old_pid = page_id_from_path(old_path, wiki_dir)

    dst = wiki_dir / "sources" / old_path.name
    new_pid = page_id_from_path(dst, wiki_dir)

    fm = page.frontmatter or {}
    if new_confidence is not None:
        fm["confidence"] = round(float(new_confidence), 2)
    if extra_frontmatter:
        fm.update(extra_frontmatter)
    title = str(fm.get("title") or dst.stem)

    # 1) Write to sources/ and remove the review/ copy.
    page.path = dst
    page.frontmatter = fm
    write_page(page)
    old_path.unlink(missing_ok=True)

    # 2) Purge the stale review-id units, then index under the new id exactly as a
    #    freshly-ingested live page would be.
    await delete_page_index(
        bm25, dense, old_pid,
        chunk_count=fm.get("chunk_count"), media_count=fm.get("media_count"),
    )
    chunk_count = await index_page_chunks(bm25, dense, new_pid, title, page.body, fm)
    await index_doc2query(bm25, dense, new_pid, title, fm.get("hypothetical_questions") or [])

    # Keep the stamped count consistent with the units just written under new_pid.
    if chunk_count != fm.get("chunk_count"):
        fm["chunk_count"] = chunk_count
        page.frontmatter = fm
        write_page(page)

    # 3) Re-point the knowledge-graph rows from the old id to the new id.
    if graph is not None:
        try:
            await graph.reassign_page_id(old_pid, new_pid)
        except Exception as e:
            log.warning("promote: graph reassign failed",
                        extra={"metadata": {"old": old_pid, "new": new_pid, "error": str(e)[:160]}})

        # 3b) Re-embed media units under the new id. Step 2 deleted the old
        # `<old>#media#<n>` vectors and reassign rewrote each embedding_id to the new
        # prefix — but the vector itself must be re-created, else media_for_entity
        # retrieval points at a dense id that no longer exists. No-op when the page
        # has no media (the common case, and always when GRAPH_MULTIMODAL_NODES=off).
        if dense is not None and hasattr(graph, "media_for_page"):
            try:
                domain = fm.get("domain", "general")
                for m in await graph.media_for_page(new_pid):
                    emb_id = m.get("embedding_id")
                    content = (m.get("content") or "").strip()
                    if not emb_id or not content:
                        continue
                    embed_text = f"{m.get('caption') or ''}\n{content}".strip()[:4000]
                    with contextlib.suppress(Exception):
                        await dense.upsert(
                            emb_id, embed_text,
                            meta={"title": title, "parent_id": new_pid,
                                  "media_kind": m.get("kind"), "domain": domain},
                        )
            except Exception as e:
                log.debug("promote: media re-embed failed",
                          extra={"metadata": {"error": str(e)[:160]}})

    try:
        rebuild_index(wiki_dir)
    except Exception as e:
        log.debug("promote: rebuild_index failed", extra={"metadata": {"error": str(e)[:120]}})

    return new_pid
