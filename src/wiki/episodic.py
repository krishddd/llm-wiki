"""Episodic memory layer (MemAgents ICLR 2026 pattern).

Two-tier memory for the wiki:

- SEMANTIC (existing): wiki/sources/, wiki/entities/ — consolidated, stable knowledge
- EPISODIC (new):     wiki/episodic/<date>.md — recent events, transient state,
                       what we just discussed in queries / ingests.

Episodic entries decay (default: pruned after 14 days unless promoted). Periodic
consolidation moves stable episodic content into the semantic wiki via the
reconciler.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

EPISODIC_RETENTION_DAYS = 14


def episodic_dir(wiki_dir: Path) -> Path:
    p = Path(wiki_dir) / "episodic"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_episode(
    wiki_dir: Path,
    *,
    kind: str,                 # "query" | "ingest" | "lint" | "review-accept" | "review-reject" | "manual"
    title: str,
    body: str,
    correlation_id: str = "",
    metadata: dict | None = None,
) -> Path:
    """Append an episodic entry to today's episodic page. One file per day."""
    d = episodic_dir(wiki_dir)
    today = date.today().isoformat()
    path = d / f"{today}.md"
    now_ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    metadata = metadata or {}
    meta_str = ""
    if metadata:
        meta_str = "  \n  ".join(f"**{k}:** {v}" for k, v in metadata.items())
        meta_str = f"\n\n  {meta_str}"

    if not path.exists():
        path.write_text(
            f"---\nkind: episodic\ndate: {today}\n---\n\n# Episodic — {today}\n\n",
            encoding="utf-8",
        )
    cid_line = f"correlation_id: `{correlation_id}`\n" if correlation_id else ""
    entry = (
        f"\n## [{now_ts}] {kind} — {title}\n"
        f"{cid_line}"
        f"{meta_str}\n\n"
        f"{body.strip()}\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return path


def list_recent_episodes(wiki_dir: Path, *, days: int = 3) -> list[dict]:
    """Return the last `days` of episodic entries as flat list of dicts."""
    d = episodic_dir(wiki_dir)
    today = date.today()
    out: list[dict] = []
    entry_re = re.compile(r"^## \[(\d\d:\d\d:\d\dZ)\] (\w[\w\-]*) — (.+?)$", re.MULTILINE)
    for offset in range(days):
        target = today - timedelta(days=offset)
        f = d / f"{target.isoformat()}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for m in entry_re.finditer(text):
            out.append({
                "date": target.isoformat(),
                "time": m.group(1),
                "kind": m.group(2),
                "title": m.group(3),
            })
    return out


def read_episodes(
    wiki_dir: Path,
    *,
    correlation_ids: list[str] | None = None,
    days: int = 14,
) -> list[dict]:
    """Read raw episodic entries with their bodies (Phase F1).

    `list_recent_episodes` returns titles only; `read_episodes` reads each
    `wiki/episodic/<date>.md`, splits on the entry headers, and includes
    the body text per entry. If `correlation_ids` is given, only entries
    whose body contains a matching `correlation_id: \`COR-...\`` line are
    returned.
    """
    d = episodic_dir(Path(wiki_dir))
    today = date.today()
    out: list[dict] = []
    entry_re = re.compile(
        r"^## \[(?P<time>\d\d:\d\d:\d\dZ)\] (?P<kind>\w[\w\-]*) — (?P<title>.+?)$",
        re.MULTILINE,
    )
    cor_set = {c.strip() for c in (correlation_ids or []) if c.strip()}
    for offset in range(days):
        target = today - timedelta(days=offset)
        f = d / f"{target.isoformat()}.md"
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        matches = list(entry_re.finditer(text))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            # Pull correlation_id if present. Permissive — accept any non-backtick
            # characters between the backticks so we don't drop entries written by
            # external tooling that uses lowercase or non-COR-prefixed IDs.
            cor_match = re.search(r"correlation_id:\s*`([^`]+)`", body)
            cid = cor_match.group(1) if cor_match else None
            if cor_set and (cid is None or cid not in cor_set):
                continue
            out.append({
                "date": target.isoformat(),
                "time": m.group("time"),
                "kind": m.group("kind"),
                "title": m.group("title"),
                "body": body,
                "correlation_id": cid,
            })
    return out


def prune_old_episodes(wiki_dir: Path, *, retention_days: int = EPISODIC_RETENTION_DAYS) -> int:
    """Delete episodic files older than `retention_days`. Returns count deleted."""
    d = episodic_dir(wiki_dir)
    cutoff = date.today() - timedelta(days=retention_days)
    deleted = 0
    for f in d.glob("*.md"):
        try:
            stem_date = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if stem_date < cutoff:
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted
