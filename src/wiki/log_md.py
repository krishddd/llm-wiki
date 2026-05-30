"""Append-only operation log. Entries start with `## [ISO-DATE] ACTION | title` so unix tools can parse."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def append_log(wiki_dir: Path, action: str, title: str, details: str = "") -> None:
    wiki_dir = Path(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    log_path = wiki_dir / "log.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    entry = f"\n## [{ts}] {action} | {title}\n"
    if details:
        entry += f"\n{details}\n"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Operation Log\n"
    log_path.write_text(existing + entry, encoding="utf-8")
