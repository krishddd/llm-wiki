"""Small-to-big chunk reconstruction (parent-document retrieval done right).

Ingest indexes pages as 1500-char sub-chunks (`<pid>#<idx>`), but retrieval used to
collapse chunk hits to the parent page and rerank/synthesise on `page_text[:4000]` —
the *first* 4000 chars — so a match beyond that horizon was retrieved and then never
shown to the reranker or the synthesiser. This module deterministically re-derives
the exact sub-chunks the indexes saw (same construction as
`Ingestor._index_page_chunks`) so hybrid search can hand the *matched* chunks (small
= precise match, big = surrounding parent context) downstream instead.
"""
from __future__ import annotations

from .contextual import merge_context_with_chunk

# Must mirror the values used at index time in `Ingestor._index_page_chunks`.
SUB_CHUNK_TARGET = 1500
SUB_CHUNK_OVERLAP = 200


def chunk_text(text: str, *, target_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Canonical sliding-window chunker — single source of truth shared with ingest."""
    if len(text) <= target_chars:
        return [text]
    chunks: list[str] = []
    i = 0
    while i < len(text):
        end = min(i + target_chars, len(text))
        chunks.append(text[i:end])
        if end >= len(text):
            break
        i = end - overlap
    return chunks


def indexable_text(title: str, body: str, frontmatter: dict | None) -> str:
    """Reconstruct the exact text `_index_page_chunks` chunked at index time."""
    text = f"{title}\n{body}"
    preamble = (frontmatter or {}).get("context_preamble")
    if preamble:
        text = merge_context_with_chunk(str(preamble), text)
    return text


def matched_chunk_indices(chunk_ids: list[str]) -> dict[str, list[int]]:
    """Map raw index hits (`pid`, `pid#3`, `pid#media#2`, `pid#hq`) → {pid: [idx, …]}.

    Only plain numeric sub-chunk suffixes carry a reconstructable offset; media/hq
    hits (and whole-page ids) still vote for their parent but contribute no index.
    """
    out: dict[str, list[int]] = {}
    for cid in chunk_ids:
        parts = cid.split("#")
        pid = parts[0]
        out.setdefault(pid, [])
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx not in out[pid]:
                out[pid].append(idx)
    return out


def focused_text(
    title: str,
    body: str,
    frontmatter: dict | None,
    chunk_idxs: list[int],
    *,
    max_chars: int = 4000,
    include_neighbors: bool = True,
) -> str:
    """Return the matched sub-chunks (small) padded with their neighbours (big),
    joined in document order and capped at `max_chars`.

    Falls back to "" when no valid index survives (caller should then use the
    page-prefix behaviour).
    """
    if not chunk_idxs:
        return ""
    chunks = chunk_text(
        indexable_text(title, body, frontmatter),
        target_chars=SUB_CHUNK_TARGET,
        overlap=SUB_CHUNK_OVERLAP,
    )
    wanted: set[int] = set()
    for i in sorted(set(chunk_idxs)):
        if 0 <= i < len(chunks):
            wanted.add(i)
            if include_neighbors:
                if i - 1 >= 0:
                    wanted.add(i - 1)
                if i + 1 < len(chunks):
                    wanted.add(i + 1)
    if not wanted:
        return ""

    parts: list[str] = []
    total = 0
    prev: int | None = None
    for i in sorted(wanted):
        piece = chunks[i]
        if prev is not None and i == prev + 1:
            # consecutive chunks overlap by SUB_CHUNK_OVERLAP — drop the duplicate seam
            piece = piece[SUB_CHUNK_OVERLAP:]
            parts[-1] = parts[-1] + piece
        else:
            if prev is not None:
                parts.append("\n[…]\n")
            parts.append(piece)
        prev = i
        total += len(piece)
        if total >= max_chars:
            break
    return "".join(parts)[:max_chars]
