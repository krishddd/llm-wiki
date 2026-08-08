"""Document-element abstraction for multimodal ingest.

Every loader returns `list[DocElement]`. Downstream code renders them to Markdown
(for embedding / the wiki page body) or inspects `.kind` to decide per-element
handling (e.g. llava-caption an image, keep a table atomic during chunking).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ElementKind = Literal["heading", "text", "table", "image", "code"]


@dataclass
class DocElement:
    kind: ElementKind
    content: str  # for "table" this is already GitHub-flavoured Markdown;
    # for "image" this is the caption / alt-text (or "" until llava fills it)
    meta: dict = field(default_factory=dict)
    # Useful metadata keys: page, section, level (for headings), path (to image file),
    # mime (image/png etc), bytes_b64 (short), ocr (bool).


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a matrix of cell strings as a Markdown table. First row treated as header."""
    if not rows:
        return ""
    # Normalise width
    width = max(len(r) for r in rows)
    rows = [[(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in r] + [""] * (width - len(r)) for r in rows]
    head = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def elements_to_markdown(elements: list[DocElement]) -> str:
    """Flatten a DocElement list to a single Markdown string (for the wiki page body / embedding)."""
    parts: list[str] = []
    for el in elements:
        if el.kind == "heading":
            lvl = int(el.meta.get("level", 2))
            parts.append(f"{'#' * max(1, min(lvl, 6))} {el.content.strip()}")
        elif el.kind == "table":
            page = el.meta.get("page")
            caption = el.meta.get("caption")
            head = f"**Table{f' (p.{page})' if page else ''}{f': {caption}' if caption else ''}**"
            parts.append(f"{head}\n\n{el.content}")
        elif el.kind == "image":
            cap = el.content.strip() or el.meta.get("alt") or "[image]"
            page = el.meta.get("page")
            parts.append(f"![{cap}]({el.meta.get('path', '#')})" + (f"  _(p.{page})_" if page else ""))
        elif el.kind == "code":
            lang = el.meta.get("lang", "")
            parts.append(f"```{lang}\n{el.content}\n```")
        else:
            t = el.content.strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def layout_aware_chunks(
    elements: list[DocElement],
    *,
    target_chars: int = 6000,
    overlap_chars: int = 200,
) -> list[str]:
    """Chunk elements into ~target_chars strings without splitting tables or images.

    Headings start a new chunk if the current chunk is already substantial. Tables
    and images are always atomic — if a table is larger than target_chars it still
    becomes a single (oversize) chunk. Overlap carries the tail of the previous
    chunk to the next as plain text.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    for el in elements:
        piece = elements_to_markdown([el])
        if not piece:
            continue
        atomic = el.kind in ("table", "image")
        if atomic and buf_len + len(piece) > target_chars and buf_len > 0:
            flush()
        if el.kind == "heading" and buf_len > target_chars * 0.5:
            flush()
        if buf_len + len(piece) > target_chars and buf_len > 0 and not atomic:
            flush()
        buf.append(piece)
        buf_len += len(piece) + 2

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:], strict=False):
            # Clamp: if a chunk is shorter than overlap_chars we'd otherwise
            # duplicate the entire previous chunk, ballooning the token count.
            n = min(overlap_chars, max(0, len(prev) // 4))
            tail = prev[-n:] if n else ""
            overlapped.append(f"{tail}\n\n{curr}" if tail else curr)
        return overlapped
    return chunks
