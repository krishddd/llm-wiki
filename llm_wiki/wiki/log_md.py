"""Operation log following the OKF v0.1 `log.md` convention.

Entries are grouped under `## YYYY-MM-DD` headings, newest date first, newest entry
first within each day, using bold action keywords:

    ## 2026-07-15

    - **Ingest** Some Document — chunks: 4 · confidence: 0.82 (14:23 UTC)

Legacy entries in the older `## [ts] ACTION | title` format may remain below the
date-grouped section; they are left untouched.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

_HEADER = "# Operation Log"
_DATE_HEADING_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def append_log(wiki_dir: Path, action: str, title: str, details: str = "") -> None:
    wiki_dir = Path(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    log_path = wiki_dir / "log.md"

    now = datetime.now(UTC)
    today = now.strftime("%Y-%m-%d")
    keyword = (action or "update").strip().replace("-", " ").title()
    entry = f"- **{keyword}** {title}"
    if details:
        entry += f" — {details}"
    entry += f" ({now.strftime('%H:%M')} UTC)"

    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else f"{_HEADER}\n"
    if _HEADER not in existing:
        existing = f"{_HEADER}\n\n{existing}"

    today_heading = f"## {today}"
    m = _DATE_HEADING_RE.search(existing)
    if m and m.group(1) == today:
        # Newest-first within the day: insert right after today's heading line.
        insert_at = existing.index("\n", m.start()) + 1
        updated = existing[:insert_at] + f"\n{entry}\n" + existing[insert_at:].lstrip("\n")
    else:
        # New (newer) date group goes directly under the file header, before
        # any older groups or legacy-format entries.
        header_end = existing.index(_HEADER) + len(_HEADER)
        block = f"\n\n{today_heading}\n\n{entry}\n"
        rest = existing[header_end:].lstrip("\n")
        updated = existing[:header_end] + block + ("\n" + rest if rest else "")

    log_path.write_text(updated, encoding="utf-8")
