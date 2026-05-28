"""Automated Page Compaction & Refactoring — A-Mem Zettelkasten Maintenance.

Compacts heavily evolved pages (>15KB or ≥3 evolution edits) by refactoring
disparate updates into a cohesive readable flow and archiving superseded
history blocks into wiki/archive/history/ to prevent RAG context-window bloat.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from ..llm import OllamaClient
from .pages import Page, read_page, write_page

log = logging.getLogger(__name__)

COMPACT_SYSTEM = (
    "You are a master wiki editor. You are refactoring a wiki page that has accumulated "
    "disparate update sections and superseded blocks over time. Unify all recent updates "
    "and edits seamlessly into the main page body so the text reads as a single cohesive, "
    "highly professional, and structured reference page.\n\n"
    "CRITICAL RULES:\n"
    "- Preserve all core factual claims, mathematical LaTeX formulas ($...$, $$...$$), and markdown tables.\n"
    "- Strip away all redundant headers, duplicated paragraphs, or inline update callouts (e.g. 'Updated from Source...'). "
    "Merge them smoothly into the main text body.\n"
    "- DO NOT preserve the historical 'Superseded' sections. We will archive those separately. Remove all '## Superseded' blocks completely.\n"
    "- Return ONLY the final, beautifully refactored markdown body text. No commentary or metadata outside the markdown."
)

async def compact_bloated_pages(
    *,
    wiki_dir: Path,
    client: OllamaClient,
    bm25=None,
    dense=None,
    size_threshold_bytes: int = 15000,
    evolve_threshold_count: int = 3,
) -> dict:
    """Scan wiki/sources/, identify bloated evolved pages, refactor them, and archive superseded text.

    Stages refined page drafts in wiki/review/compact/ for safety.
    """
    sources_dir = Path(wiki_dir) / "sources"
    history_dir = Path(wiki_dir) / "archive" / "history"
    review_dir = Path(wiki_dir) / "review" / "compact"
    
    history_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    if not sources_dir.exists():
        return {"scanned": 0, "compacted": 0, "staged_pages": []}

    scanned = 0
    compacted = 0
    staged_pages = []

    for path in sorted(sources_dir.glob("*.md")):
        scanned += 1
        try:
            page = read_page(path)
        except Exception:
            continue

        fm = page.frontmatter or {}
        evolved_count = len(fm.get("evolved_by", []) or [])
        file_size = path.stat().st_size

        # Gate compaction on size OR evolution counts
        if file_size < size_threshold_bytes and evolved_count < evolve_threshold_count:
            continue

        log.info(
            "page qualified for compaction",
            extra={"metadata": {"page": path.name, "size_kb": round(file_size / 1024, 2), "edits": evolved_count}},
        )

        # 1) Separate the core body from any ## Superseded blocks
        body = page.body or ""
        superseded_blocks = []
        
        # Regex to locate ## Superseded sections
        split_parts = re.split(r"(##\s+Superseded\s+\(.*?\).*?)(?=##\s+|$)", body, flags=re.DOTALL | re.IGNORECASE)
        core_body_parts = []
        for part in split_parts:
            if part.strip().lower().startswith("## superseded"):
                superseded_blocks.append(part.strip())
            else:
                core_body_parts.append(part)
        
        core_body = "\n\n".join(core_body_parts).strip()

        # 2) Ask Qwen to compact the core body seamlessly
        prompt = (
            f"PAGE TITLE: {fm.get('title', path.stem)}\n\n"
            f"PAGE BODY WITH DISPARATE UPDATES:\n\n"
            f"{core_body}\n\n"
            f"Refactor the body into a unified clean layout."
        )

        try:
            compacted_body = await client.qwen(prompt, system=COMPACT_SYSTEM, temperature=0.1)
            compacted_body = compacted_body.strip() if compacted_body else core_body
        except Exception as e:
            log.warning("compaction LLM call failed", extra={"metadata": {"page": path.name, "error": str(e)[:160]}})
            continue

        # 3) Archive superseded blocks if they exist
        archive_path = None
        if superseded_blocks:
            archive_name = f"{path.stem}-history.md"
            archive_path = history_dir / archive_name
            today = date.today().isoformat()
            
            archive_content = (
                "---\n"
                f"title: \"{fm.get('title', path.stem)} - Historical Revisions\"\n"
                f"parent_page: \"{path.name}\"\n"
                f"archived_on: \"{today}\"\n"
                "---\n\n"
                f"# History Archive for `{fm.get('title', path.stem)}`\n\n"
                + "\n\n---\n\n".join(superseded_blocks)
            )
            try:
                archive_path.write_text(archive_content, encoding="utf-8")
            except Exception as e:
                log.warning("writing history archive failed", extra={"metadata": {"page": path.name, "error": str(e)[:160]}})

        # 4) Append link to history archive inside the compacted page
        if archive_path:
            rel_path = f"../archive/history/{path.stem}-history.md"
            compacted_body += (
                f"\n\n---\n\n"
                f"> [!note] Historical Revisions Archive\n"
                f"> [Detailed superseded revisions of this page can be viewed in the history archive]({rel_path}).\n"
            )

        # 5) Stage the compacted page under review/compact/ for safety
        staged_path = review_dir / path.name
        new_fm = dict(fm)
        new_fm["compacted_on"] = date.today().isoformat()
        new_fm["original_size_bytes"] = file_size
        new_fm["compacted_size_bytes"] = len(compacted_body.encode("utf-8"))

        try:
            write_page(Page(path=staged_path, frontmatter=new_fm, body=compacted_body))
            compacted += 1
            staged_pages.append(path.name)
            log.info("staged compacted page successfully", extra={"metadata": {"staged_path": str(staged_path)}})
        except Exception as e:
            log.warning("writing staged page failed", extra={"metadata": {"page": path.name, "error": str(e)[:160]}})

    return {
        "scanned": scanned,
        "compacted": compacted,
        "staged_pages": staged_pages
    }
