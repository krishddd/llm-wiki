"""Parse a Markdown answer into typed AnswerBlocks for NotebookLM-style rendering.

Block kinds:
- heading   ({"level": 1-6})
- text      (plain paragraph; may contain inline citations [1] [2])
- list      ({"ordered": bool, "items": [str]})
- table     (raw GFM markdown table)
- code      ({"lang": str})
- math      ({"display": bool})  — display:True for $$ ... $$, False for $ ... $
- quote     (blockquote text)
- callout   ({"kind": "note"|"warning"|"tip"|"info"})

Also: convert `[Page Title]` citation tokens to numbered `[1]` citations, and emit
the citation index as a list aligned by appearance order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "text", "list", "table", "code", "math", "quote", "callout"]


@dataclass
class AnswerBlock:
    kind: BlockKind
    content: str
    meta: dict = field(default_factory=dict)


# ───── citation numbering ─────

_CITE_RE = re.compile(r"\[([^\[\]]{2,100}?)\](?!\()")
# Captures bracketed tokens that are NOT markdown links `[text](url)`.

def number_citations(answer: str, citation_titles: list[str]) -> tuple[str, list[int]]:
    """Replace `[Page Title]` tokens with `[1]`, `[2]`, … numbered in first-appearance order.

    Returns (rewritten_answer, used_indices_in_appearance_order).
    `used_indices` are 1-based positions into `citation_titles`.
    """
    if not citation_titles:
        return answer, []

    titles_low = [t.strip().lower() for t in citation_titles]
    appearance: list[int] = []        # citation_titles indices in first-seen order (0-based)
    appearance_set: set[int] = set()
    cite_to_appearance_num: dict[int, int] = {}  # 0-based citation idx -> 1-based appearance num

    def _resolve(token: str) -> int | None:
        t = token.strip().lower()
        # Exact match first
        for i, ct in enumerate(titles_low):
            if t == ct:
                return i
        # Substring overlap fallback
        for i, ct in enumerate(titles_low):
            if t in ct or ct in t:
                return i
        return None

    def _sub(m: re.Match) -> str:
        token = m.group(1)
        # Skip footnote-style or already-numbered tokens
        if token.startswith("^") or re.fullmatch(r"\d+(,\s*\d+)*", token):
            return m.group(0)
        idx = _resolve(token)
        if idx is None:
            return m.group(0)  # leave as-is if we can't map it
        if idx not in appearance_set:
            appearance.append(idx)
            appearance_set.add(idx)
            cite_to_appearance_num[idx] = len(appearance)
        return f"[{cite_to_appearance_num[idx]}]"

    rewritten = _CITE_RE.sub(_sub, answer)
    return rewritten, [i + 1 for i in range(len(appearance))]  # 1..N appearance order


# ───── markdown → typed blocks ─────

_TABLE_RE = re.compile(
    r"(?ms)^\|[^\n]+\|\s*\n\|[\s:\-\|]+\|\s*\n(?:\|[^\n]*\|\s*\n?)+"
)
_CODE_RE = re.compile(r"(?ms)^```(\w*)\s*\n(.*?)\n```")
_MATH_DISPLAY_RE = re.compile(r"(?ms)^\$\$\s*\n?(.*?)\n?\$\$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_OL_LINE = re.compile(r"^\s*\d+\.\s+(.+)$")
_UL_LINE = re.compile(r"^\s*[-*+]\s+(.+)$")
_QUOTE_LINE = re.compile(r"^>\s?(.*)$")
_CALLOUT_RE = re.compile(r"^>\s*\[!(\w+)\]\s*(.*)$", re.IGNORECASE)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^\$\n]{1,400})\$(?!\$)")


def parse_blocks(answer: str) -> list[AnswerBlock]:
    """Walk the answer and emit typed blocks. Robust to common LLM-Markdown quirks."""
    if not answer or not answer.strip():
        return []

    blocks: list[AnswerBlock] = []
    placeholders: list[AnswerBlock] = []

    def _stash(b: AnswerBlock) -> str:
        idx = len(placeholders)
        placeholders.append(b)
        return f"\x00BLK{idx}\x00"

    # 1) Pull out fenced code blocks first (they may contain anything)
    text = _CODE_RE.sub(
        lambda m: _stash(AnswerBlock(kind="code", content=m.group(2), meta={"lang": m.group(1) or ""})),
        answer,
    )
    # 2) Display math
    text = _MATH_DISPLAY_RE.sub(
        lambda m: _stash(AnswerBlock(kind="math", content=m.group(1).strip(), meta={"display": True})),
        text,
    )
    # 3) Tables
    text = _TABLE_RE.sub(
        lambda m: _stash(AnswerBlock(kind="table", content=m.group(0).strip())),
        text,
    )

    # 4) Walk line-by-line for headings, lists, quotes, paragraphs
    lines = text.split("\n")
    i = 0
    paragraph_buf: list[str] = []
    list_buf: list[str] = []
    list_ordered: bool | None = None
    quote_buf: list[str] = []
    callout_kind: str | None = None

    def flush_para() -> None:
        nonlocal paragraph_buf
        if paragraph_buf:
            content = "\n".join(paragraph_buf).strip()
            if content:
                # Replace placeholders embedded inside paragraphs by emitting them
                # in document order alongside the paragraph text.
                _emit_with_placeholders(content)
            paragraph_buf = []

    def flush_list() -> None:
        nonlocal list_buf, list_ordered
        if list_buf:
            blocks.append(
                AnswerBlock(
                    kind="list",
                    content="\n".join(list_buf),
                    meta={"ordered": bool(list_ordered), "items": list_buf[:]},
                )
            )
            list_buf = []
            list_ordered = None

    def flush_quote() -> None:
        nonlocal quote_buf, callout_kind
        if quote_buf:
            content = "\n".join(quote_buf).strip()
            if content:
                if callout_kind:
                    blocks.append(AnswerBlock(kind="callout", content=content, meta={"kind": callout_kind.lower()}))
                else:
                    blocks.append(AnswerBlock(kind="quote", content=content))
        quote_buf = []
        callout_kind = None

    def _emit_with_placeholders(content: str) -> None:
        """A paragraph's text may be interrupted by stashed placeholders. Emit them
        in document order, splitting the surrounding text into text-blocks."""
        parts = re.split(r"\x00BLK(\d+)\x00", content)
        for i, p in enumerate(parts):
            if i % 2 == 0:
                t = p.strip()
                if t:
                    blocks.append(AnswerBlock(kind="text", content=t))
            else:
                blocks.append(placeholders[int(p)])

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line — paragraph break
        if not stripped:
            flush_para()
            flush_list()
            flush_quote()
            i += 1
            continue

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            flush_para(); flush_list(); flush_quote()
            blocks.append(AnswerBlock(kind="heading", content=m.group(2).strip(), meta={"level": len(m.group(1))}))
            i += 1
            continue

        # Callout (Obsidian-style)
        m = _CALLOUT_RE.match(line)
        if m:
            flush_para(); flush_list(); flush_quote()
            callout_kind = m.group(1)
            quote_buf.append(m.group(2))
            i += 1
            continue

        # Quote
        m = _QUOTE_LINE.match(line)
        if m:
            flush_para(); flush_list()
            quote_buf.append(m.group(1))
            i += 1
            continue

        # Ordered list
        m = _OL_LINE.match(line)
        if m:
            flush_para(); flush_quote()
            if list_ordered is False:
                flush_list()
            list_ordered = True
            list_buf.append(m.group(1).strip())
            i += 1
            continue

        # Unordered list
        m = _UL_LINE.match(line)
        if m:
            flush_para(); flush_quote()
            if list_ordered is True:
                flush_list()
            list_ordered = False
            list_buf.append(m.group(1).strip())
            i += 1
            continue

        # Plain paragraph line
        flush_list(); flush_quote()
        paragraph_buf.append(line)
        i += 1

    flush_para(); flush_list(); flush_quote()
    return blocks
