"""Extract multimodal blocks (tables, images, code) from a Markdown wiki page body
and rank them against a query so the most relevant ones can be surfaced in the
query response alongside the textual snippet.

Pages produced by the upgraded ingest pipeline carry a `## Tables & Figures`
section after the summary; this module is robust to body shapes that don't have
that section either — it scans the whole body.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ExcerptKind = Literal["table", "image", "code"]


@dataclass
class MultimodalExcerpt:
    kind: ExcerptKind
    content: str            # markdown source of the block (renderable as-is)
    meta: dict = field(default_factory=dict)
    score: float = 0.0      # lexical-overlap score against the query


# A GFM pipe-table: header line | --- separator | body row(s)
_TABLE_RE = re.compile(
    r"(?ms)^(?:\*\*Table[^\n]*\*\*\s*\n+)?"      # optional bold "Table" caption line
    r"(\|[^\n]+\|\s*\n"                          # header
    r"\|[\s:\-\|]+\|\s*\n"                       # separator
    r"(?:\|[^\n]*\|\s*\n?)+)"                    # body rows
)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)(?:\s*_\(p\.(\d+)\)_)?")
_CODE_RE = re.compile(r"(?ms)^```(\w*)\s*\n(.*?)\n```")


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s) if len(t) > 2}


def _score(block: str, query_toks: set[str]) -> float:
    if not query_toks:
        return 0.0
    block_toks = _tokens(block)
    if not block_toks:
        return 0.0
    inter = len(query_toks & block_toks)
    return inter / (len(query_toks) ** 0.5)  # mild downweight for very long queries


def extract_excerpts(body: str, query: str | None = None, *, max_per_kind: int = 2) -> list[MultimodalExcerpt]:
    """Parse a page body, return up to `max_per_kind` of each kind, sorted by relevance to `query`.

    If `query` is None, returns blocks in document order with score 0.
    """
    excerpts: list[MultimodalExcerpt] = []

    # Tables — keep optional preceding "**Table ...**" caption line if present.
    for m in _TABLE_RE.finditer(body):
        block = m.group(0).strip()
        # Capture caption from preceding line if there is one
        caption = ""
        cap_m = re.match(r"^\*\*(Table[^*]+)\*\*", block)
        if cap_m:
            caption = cap_m.group(1).strip()
        excerpts.append(MultimodalExcerpt(kind="table", content=block, meta={"caption": caption}))

    # Images — content is the alt-text caption; meta carries the path.
    for m in _IMAGE_RE.finditer(body):
        alt, path, page = m.group(1), m.group(2), m.group(3)
        if not alt and not path:
            continue
        excerpts.append(
            MultimodalExcerpt(
                kind="image",
                content=m.group(0),
                meta={"alt": alt, "path": path, "page": page},
            )
        )

    # Code blocks — skip empty ones; restrict to >= 20 chars to avoid trivial inline fences.
    for m in _CODE_RE.finditer(body):
        lang, code = m.group(1) or "", m.group(2).strip()
        if len(code) < 20:
            continue
        excerpts.append(
            MultimodalExcerpt(
                kind="code",
                content=f"```{lang}\n{code}\n```",
                meta={"lang": lang},
            )
        )

    if not excerpts:
        return []

    # Score against query if provided
    qtoks = _tokens(query or "")
    if qtoks:
        for e in excerpts:
            e.score = _score(e.content, qtoks)
        excerpts.sort(key=lambda e: e.score, reverse=True)

    # Cap per kind to avoid one huge table dominating a citation
    out: list[MultimodalExcerpt] = []
    counts: dict[str, int] = {"table": 0, "image": 0, "code": 0}
    for e in excerpts:
        if counts[e.kind] >= max_per_kind:
            continue
        # Only keep zero-score items if we have nothing better; require >0 when a query was given.
        if qtoks and e.score == 0 and out:
            continue
        out.append(e)
        counts[e.kind] += 1
    return out


def truncate_block(content: str, kind: ExcerptKind, max_chars: int = 1200) -> str:
    """Trim a block for response payload size. Tables: keep header + separator + first N rows.
    Images: untouched (already short). Code: truncate with marker."""
    if len(content) <= max_chars:
        return content
    if kind == "table":
        lines = content.splitlines()
        # Find separator line index
        sep_idx = next((i for i, ln in enumerate(lines) if re.match(r"^\|[\s:\-\|]+\|\s*$", ln)), 1)
        kept = lines[: sep_idx + 1]
        budget = max_chars - sum(len(l) + 1 for l in kept)
        for ln in lines[sep_idx + 1 :]:
            if budget - len(ln) - 1 < 60:
                break
            kept.append(ln)
            budget -= len(ln) + 1
        kept.append("| ... | _(truncated)_ |")
        return "\n".join(kept)
    if kind == "code":
        body = content.strip("`").split("\n", 1)[-1]
        return content[:max_chars] + "\n... (truncated) ...\n```"
    return content[:max_chars]
