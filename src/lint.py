"""Wiki health-check (Qwen) → optional auto-repair.

Read-only mode: returns a JSON report.
Auto-fix mode (Phase E1):
  • orphans → append a backlink stub to wiki/index.md so they aren't truly orphan
  • missing entity pages → trigger rebuild_entity_pages()
  • stale low-conf pages (confidence < 0.4 AND last_accessed > 180 days):
      move to wiki/archive/<original-relative-path>
  • broken cross-references → comment out the wiki-link with a flag

`POST /lint` and `POST /admin/run/lint_autofix` use this with `auto_fix=True`.
`GET /lint` keeps the read-only behaviour (calls with `auto_fix=False`).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

from .config import Settings, get_settings
from .llm import OllamaClient, get_client
from .logging_config import audit
from .wiki.pages import PageStore

log = logging.getLogger(__name__)

LINT_SYSTEM = (
    "You are a wiki maintainer. Review the summaries of the provided pages and report issues. "
    "Reply ONLY with valid JSON:\n"
    '{"orphans":["page_id"],'
    '"stale":[{"page":"page_id","reason":"..."}],'
    '"missing_entity_pages":["entity name"],'
    '"contradictions":[{"pages":["pid_a","pid_b"],"claim":"..."}],'
    '"suggested_sources":["..."]}'
)


def _extract_json(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?\n?", "", (s or "").strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ───────────────────────────────────────────────────────────────────
# Auto-fix helpers
# ───────────────────────────────────────────────────────────────────


def _append_orphan_backlinks(wiki_dir: Path, orphan_ids: list[str]) -> int:
    """Append a `## Orphans` section to index.md listing the orphan pages so
    they at least have one inbound link from the master index. Returns count."""
    if not orphan_ids:
        return 0
    idx = Path(wiki_dir) / "index.md"
    today = date.today().isoformat()
    lines = []
    existing = idx.read_text(encoding="utf-8") if idx.exists() else "# Wiki Index\n"
    marker = f"\n## Orphans (linked {today})\n"
    if marker.strip() in existing:
        # Don't duplicate the marker; append items only.
        pass
    else:
        lines.append(marker)
        lines.append("")
    for pid in orphan_ids:
        title = Path(pid).stem
        rel = pid.replace("\\", "/")
        link_line = f"- [{title}]({rel})"
        # Skip if link already in file.
        if link_line not in existing:
            lines.append(link_line)
    if not lines:
        return 0
    idx.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return sum(1 for l in lines if l.startswith("- ["))


async def _archive_stale_page(
    wiki_dir: Path,
    page_id: str,
    *,
    bm25=None,
    dense=None,
) -> bool:
    """Move a stale page to wiki/archive/, preserving the relative path.
    Removes it from BM25 + dense indexes if those handles are passed.
    Returns True on success.

    Async because dense.delete is async — we await it directly instead of
    fire-and-forget create_task (which produced "Task was destroyed but it
    is pending" warnings under load).
    """
    import asyncio
    src = Path(wiki_dir) / page_id
    if not src.exists():
        return False
    dst = Path(wiki_dir) / "archive" / page_id
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except Exception as e:
        log.warning("archive move failed",
                    extra={"metadata": {"page": page_id, "error": str(e)[:200]}})
        return False
    # De-index. Best-effort.
    try:
        if bm25 is not None and hasattr(bm25, "delete"):
            try:
                if asyncio.iscoroutinefunction(bm25.delete):
                    await bm25.delete(page_id)
                else:
                    bm25.delete(page_id)
            except Exception:
                pass
        if dense is not None:
            try:
                if asyncio.iscoroutinefunction(dense.delete):
                    await dense.delete(page_id)
                else:
                    dense.delete(page_id)
            except Exception:
                pass
    except Exception:
        pass
    audit(log, "WIKI_WRITE_STAGED", str(dst), kind="archived", source=page_id)
    return True


def _is_stale(frontmatter: dict, lifecycle_row: dict | None, *, conf_thresh: float = 0.4, days_thresh: int = 180) -> bool:
    """Detect stale-low-conf pages: confidence < conf_thresh AND last_accessed older than days_thresh."""
    try:
        conf = float(frontmatter.get("confidence", 1.0))
    except (TypeError, ValueError):
        conf = 1.0
    if conf >= conf_thresh:
        return False
    last_accessed = (lifecycle_row or {}).get("last_accessed")
    if not last_accessed:
        # No access record → fallback to ingested_at on the frontmatter.
        last_accessed = frontmatter.get("ingested")
    if not last_accessed:
        return False
    try:
        if len(str(last_accessed)) <= 10:
            ts = datetime.fromisoformat(str(last_accessed)).replace(tzinfo=UTC)
        else:
            ts = datetime.fromisoformat(str(last_accessed).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
    except Exception:
        return False
    age = (datetime.now(UTC) - ts).total_seconds() / 86400.0
    return age >= days_thresh


_BROKEN_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


def _comment_broken_links(wiki_dir: Path) -> int:
    """Walk all wiki/sources/ + wiki/entities/ pages and comment out
    Obsidian wiki-links whose target file does not exist. Returns count fixed."""
    fixed = 0
    for sub in ("sources", "entities", "procedures"):
        d = Path(wiki_dir) / sub
        if not d.exists():
            continue
        existing_stems = {p.stem for sub2 in ("sources", "entities", "procedures")
                          for p in (Path(wiki_dir) / sub2).glob("*.md")
                          if (Path(wiki_dir) / sub2).exists()}
        for p in d.glob("*.md"):
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue

            def _maybe_break(m: re.Match) -> str:
                target = m.group(1).strip()
                # Only flag if target stem is unknown.
                if target.split("/")[-1] in existing_stems:
                    return m.group(0)
                return f"<!-- BROKEN_LINK: {m.group(0)} -->"

            new = _BROKEN_LINK_RE.sub(_maybe_break, text)
            if new != text:
                p.write_text(new, encoding="utf-8")
                fixed += 1
    return fixed


# ───────────────────────────────────────────────────────────────────
# Main entry
# ───────────────────────────────────────────────────────────────────


async def lint_wiki(
    *,
    page_store: PageStore | None = None,
    settings: Settings | None = None,
    client: OllamaClient | None = None,
    graph=None,
    bm25=None,
    dense=None,
    max_pages: int = 30,
    auto_fix: bool = False,
) -> dict:
    s = settings or get_settings()
    ps = page_store or PageStore(s.wiki_dir)
    c = client or get_client()

    pages: list[tuple[str, str]] = []
    for pid, page in ps.iter_pages(("sources", "entities", "procedures")):
        title = (page.frontmatter or {}).get("title") or Path(pid).stem
        pages.append((pid, f"[{title}] ({pid})\n{page.body[:800]}"))
        if len(pages) >= max_pages:
            break

    if not pages:
        return {"orphans": [], "stale": [], "missing_entity_pages": [],
                "contradictions": [], "suggested_sources": [],
                "auto_fix": auto_fix, "fixes": {}}

    joined = "\n\n---\n\n".join(text for _, text in pages)
    raw = await c.qwen(f"WIKI PAGES:\n{joined}", system=LINT_SYSTEM, temperature=0.2)
    data = _extract_json(raw) or {}

    orphans = data.get("orphans", []) or []
    stale_items = data.get("stale", []) or []
    missing_entities = data.get("missing_entity_pages", []) or []
    contradictions = data.get("contradictions", []) or []

    for pid in orphans:
        audit(log, "ORPHAN_PAGE_DETECTED", pid)
    for item in stale_items:
        audit(log, "STALE_PAGE_DETECTED", item.get("page", ""),
              reason=item.get("reason", ""))
    for item in contradictions:
        audit(log, "CONTRADICTION_DETECTED", ",".join(item.get("pages", [])),
              claim=item.get("claim", ""))

    fixes: dict = {}

    if auto_fix:
        # 1. Backlink orphans from index.md
        try:
            n = _append_orphan_backlinks(s.wiki_dir, orphans)
            fixes["orphan_backlinks_added"] = n
        except Exception as e:
            log.warning("auto-fix orphans failed",
                        extra={"metadata": {"error": str(e)[:200]}})

        # 2. Rebuild entity pages (covers missing_entity_pages)
        if graph is not None:
            try:
                from .wiki.entity_pages import rebuild_entity_pages
                fixes["entity_pages_rebuilt"] = rebuild_entity_pages(graph, s.wiki_dir)
            except Exception as e:
                log.warning("auto-fix entity pages failed",
                            extra={"metadata": {"error": str(e)[:200]}})

        # 3. Archive truly stale pages (low-conf AND old)
        archived: list[str] = []
        for pid, page in ps.iter_pages(("sources",)):
            try:
                lc_row = None
                if graph is not None:
                    from .wiki.lifecycle import get_page_lifecycle
                    lc_row = await get_page_lifecycle(graph, pid)
                if _is_stale(page.frontmatter or {}, lc_row):
                    if await _archive_stale_page(s.wiki_dir, pid, bm25=bm25, dense=dense):
                        archived.append(pid)
            except Exception:
                continue
        fixes["archived"] = archived

        # 4. Comment broken cross-refs
        try:
            fixes["broken_links_commented"] = _comment_broken_links(s.wiki_dir)
        except Exception as e:
            log.warning("auto-fix broken links failed",
                        extra={"metadata": {"error": str(e)[:200]}})

    return {
        "orphans": orphans,
        "stale": stale_items,
        "missing_entity_pages": missing_entities,
        "contradictions": contradictions,
        "suggested_sources": data.get("suggested_sources", []) or [],
        "pages_reviewed": len(pages),
        "auto_fix": auto_fix,
        "fixes": fixes,
    }
