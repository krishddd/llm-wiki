"""Shared page (re)indexing primitives.

Single source of truth for how a live page's text is written into the BM25 and
dense indexes, so ingest AND the review-promotion path index pages *identically*:
small-to-big sub-chunks, contextual preamble, Doc2Query `#hq` unit, and the dense
metadata (`domain`, `confidence`, `parent_id`) that retrieval depends on.

Before this module existed, `review_autopilot._accept` and the `/review/accept`
endpoint indexed a promoted page as one monolithic `index.upsert(pid, text)` blob —
losing all of the above and leaving the page a second-class retrieval citizen.

`delete_page_index` removes every retrieval unit belonging to a page id
(parent, `#hq`, `#<n>` sub-chunks, `#media#<n>` embeddings). It reads the exact
counts from frontmatter (`chunk_count` / `media_count`, stamped at index time) so a
document with >50 sub-chunks or many media nodes is cleaned completely on re-ingest,
falling back to a generous scan for pages written before those counts were recorded.
"""
from __future__ import annotations

import contextlib
import logging

from ..search.chunks import SUB_CHUNK_OVERLAP, SUB_CHUNK_TARGET, chunk_text

log = logging.getLogger(__name__)

# Fallback scan bound for pages that predate the stamped chunk/media counts.
_LEGACY_SCAN = 400


async def delete_page_index(
    bm25,
    dense,
    pid: str,
    *,
    chunk_count: int | None = None,
    media_count: int | None = None,
) -> None:
    """Delete every retrieval unit belonging to `pid` from both indexes.

    Removes the parent id, the `#hq` Doc2Query unit, the `#<n>` sub-chunks, and the
    `#media#<n>` embeddings. Exact when `chunk_count` / `media_count` are supplied
    (from frontmatter); otherwise scans a generous legacy bound.
    """
    n_chunks = _LEGACY_SCAN if chunk_count is None else max(int(chunk_count), 0)
    n_media = _LEGACY_SCAN if media_count is None else max(int(media_count), 0)
    ids = [pid, f"{pid}#hq"]
    ids += [f"{pid}#{i}" for i in range(n_chunks)]
    ids += [f"{pid}#media#{i}" for i in range(n_media)]
    for index in (bm25, dense):
        if index is None:
            continue
        for uid in ids:
            # A missing id is fine — delete is idempotent-by-intent here.
            with contextlib.suppress(Exception):
                await index.delete(uid)


async def index_page_chunks(
    bm25,
    dense,
    pid: str,
    title: str,
    body: str,
    frontmatter: dict,
) -> int:
    """Index a page's body as small-to-big sub-chunks. Returns the chunk count.

    Mirrors the indexing every live page receives at ingest. Callers should stamp
    the returned count into `frontmatter["chunk_count"]` so a later re-index / delete
    can clean up exactly.
    """
    text_to_index = f"{title}\n{body}"
    if frontmatter.get("context_preamble"):
        from ..search.contextual import merge_context_with_chunk
        text_to_index = merge_context_with_chunk(frontmatter["context_preamble"], text_to_index)

    sub_chunks = chunk_text(text_to_index, target_chars=SUB_CHUNK_TARGET, overlap=SUB_CHUNK_OVERLAP)
    for idx, ch in enumerate(sub_chunks):
        chunk_id = f"{pid}#{idx}"
        if bm25:
            try:
                await bm25.upsert(chunk_id, ch)
            except Exception as e:
                log.warning("BM25 chunk upsert failed", extra={"metadata": {"error": str(e)[:160]}})
        if dense:
            try:
                await dense.upsert(
                    chunk_id, ch,
                    meta={
                        "title": title,
                        "confidence": frontmatter.get("confidence", 0.6),
                        "parent_id": pid,
                        "domain": frontmatter.get("domain", "general"),
                    },
                )
            except Exception as e:
                log.warning("Dense chunk upsert failed", extra={"metadata": {"error": str(e)[:160]}})
    return len(sub_chunks)


async def index_doc2query(bm25, dense, pid: str, title: str, questions: list[str]) -> None:
    """Index the hypothetical questions as their own retrieval unit `<pid>#hq`."""
    if not questions:
        return
    hq_text = f"{title}\n" + "\n".join(questions)
    hq_id = f"{pid}#hq"
    if bm25:
        try:
            await bm25.upsert(hq_id, hq_text)
        except Exception as e:
            log.debug("doc2query BM25 upsert failed", extra={"metadata": {"error": str(e)[:120]}})
    if dense:
        try:
            await dense.upsert(hq_id, hq_text, meta={"title": title, "parent_id": pid, "hq": True})
        except Exception as e:
            log.debug("doc2query dense upsert failed", extra={"metadata": {"error": str(e)[:120]}})
