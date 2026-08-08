"""Markdown / plaintext loader. Splits on headings so downstream chunker can respect structure."""
from __future__ import annotations

import re
from pathlib import Path

from .elements import DocElement, elements_to_markdown

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(\w*)\s*\n(.*?)\n```", re.DOTALL | re.MULTILINE)
# A rough GFM pipe-table detector: a line starting/ending with | plus a --- row
_TABLE_RE = re.compile(r"(?m)^\|.*\|\s*\n\|[\s:\-\|]+\|\s*\n(?:\|.*\|\s*\n?)+")


def load_md_elements(path: Path) -> list[DocElement]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    elements: list[DocElement] = []

    # First extract fenced code + tables as atomic blocks, replace with placeholders,
    # then split the remainder on headings.
    placeholders: list[DocElement] = []

    def _stash(m: re.Match, kind: str) -> str:
        idx = len(placeholders)
        if kind == "code":
            placeholders.append(DocElement(kind="code", content=m.group(2), meta={"lang": m.group(1) or ""}))
        else:
            placeholders.append(DocElement(kind="table", content=m.group(0).strip()))
        return f"\x00PLACEHOLDER{idx}\x00"

    text2 = _FENCE_RE.sub(lambda m: _stash(m, "code"), text)
    text2 = _TABLE_RE.sub(lambda m: _stash(m, "table"), text2)

    # Split on headings
    pieces: list[tuple[int | None, str]] = []  # (level, content)
    last = 0
    current_head: tuple[int, str] | None = None

    for m in _HEADING_RE.finditer(text2):
        before = text2[last : m.start()].strip()
        if before:
            pieces.append((current_head[0] if current_head else None, before))
        current_head = (len(m.group(1)), m.group(2).strip())
        pieces.append(("H", current_head))  # type: ignore[arg-type]
        last = m.end()
    tail = text2[last:].strip()
    if tail:
        pieces.append((current_head[0] if current_head else None, tail))

    def _expand(s: str) -> list[DocElement]:
        out: list[DocElement] = []
        parts = re.split(r"\x00PLACEHOLDER(\d+)\x00", s)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                t = part.strip()
                if t:
                    out.append(DocElement(kind="text", content=t))
            else:
                out.append(placeholders[int(part)])
        return out

    if not pieces:
        return _expand(text2)

    for tag, payload in pieces:
        if tag == "H":
            lvl, title = payload  # type: ignore[misc]
            elements.append(DocElement(kind="heading", content=title, meta={"level": lvl}))
        else:
            elements.extend(_expand(payload))  # type: ignore[arg-type]
    return elements


def load_md(path: Path) -> str:
    return elements_to_markdown(load_md_elements(path))
