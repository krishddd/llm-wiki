"""Per-entity Markdown pages in `wiki/entities/`.

For each canonical entity with ≥ MIN_BACKLINKS pages citing it, emit
`wiki/entities/<type>-<slug>.md` with:

- YAML frontmatter: canonical_id, entity_type, aliases, backlink_count, updated
- Body: bulleted list of backlinks (pages citing this entity) + related entities
  (graph neighbours within 1 hop).

Regenerated on every ingest + review-accept. Cheap: pure SQL reads, no LLM calls.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import date
from pathlib import Path

from .pages import Page, write_page

log = logging.getLogger(__name__)

MIN_BACKLINKS = 1  # emit a page even for singleton entities — helps Obsidian graph view


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s.strip().lower()).strip("-")
    return s[:80] or "entity"


def _fetch_canonical_entities(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Return [(canonical_id, canonical_name, type)] — one row per canonical entity cluster."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id, e.name, e.type
        FROM entities e
        WHERE e.canonical_id IS NULL
        """
    )
    return [(int(r[0]), str(r[1]), str(r[2])) for r in cur.fetchall()]


def _fetch_aliases(conn: sqlite3.Connection, canonical_id: int) -> list[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM entities WHERE canonical_id = ? ORDER BY name", (canonical_id,))
    return [str(r[0]) for r in cur.fetchall()]


def _fetch_backlinks(conn: sqlite3.Connection, canonical_id: int) -> list[str]:
    """All page_ids that cite any entity row pointing (directly or via canonical_id) at this cluster."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT pe.page_id
        FROM page_entities pe
        JOIN entities e ON pe.entity_id = e.id
        WHERE e.id = ? OR e.canonical_id = ?
        ORDER BY pe.page_id
        """,
        (canonical_id, canonical_id),
    )
    return [str(r[0]) for r in cur.fetchall()]


def _fetch_related(conn: sqlite3.Connection, canonical_id: int, limit: int = 20) -> list[tuple[str, str, str]]:
    """Return [(related_name, related_type, rel_type)] — direct relations, either direction.

    Resolves both sides to canonical form.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.rel_type,
               COALESCE(e_dst.canonical_id, e_dst.id) AS dst_canon,
               e_dst.name AS dst_name,
               e_dst.type AS dst_type
        FROM relations r
        JOIN entities e_src ON r.src = e_src.id
        JOIN entities e_dst ON r.dst = e_dst.id
        WHERE COALESCE(e_src.canonical_id, e_src.id) = ?
          AND COALESCE(e_dst.canonical_id, e_dst.id) != ?
        UNION
        SELECT r.rel_type,
               COALESCE(e_src.canonical_id, e_src.id) AS src_canon,
               e_src.name AS src_name,
               e_src.type AS src_type
        FROM relations r
        JOIN entities e_src ON r.src = e_src.id
        JOIN entities e_dst ON r.dst = e_dst.id
        WHERE COALESCE(e_dst.canonical_id, e_dst.id) = ?
          AND COALESCE(e_src.canonical_id, e_src.id) != ?
        LIMIT ?
        """,
        (canonical_id, canonical_id, canonical_id, canonical_id, limit),
    )
    out: list[tuple[str, str, str]] = []
    for row in cur.fetchall():
        rel_type = str(row[0])
        name = str(row[2])
        typ = str(row[3])
        out.append((name, typ, rel_type))
    return out


def _lookup_page_title(wiki_dir: Path, page_id: str) -> str:
    p = wiki_dir / page_id
    if not p.exists():
        return page_id
    try:
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^title:\s*(.+?)$", text, flags=re.MULTILINE)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    except Exception:
        pass
    return page_id


def rebuild_entity_pages(graph, wiki_dir: Path, min_backlinks: int = MIN_BACKLINKS) -> int:
    """Write one Markdown page per canonical entity. Returns the count written."""
    from ..wiki.pages import _slug as page_slug  # noqa: F401  (ensure module loaded)

    entities_dir = Path(wiki_dir) / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    conn: sqlite3.Connection = graph._conn  # internal handle — module lives in-process
    canonicals = _fetch_canonical_entities(conn)
    written = 0
    seen_slugs: set[str] = set()

    for canonical_id, canonical_name, type_ in canonicals:
        backlinks = _fetch_backlinks(conn, canonical_id)
        if len(backlinks) < min_backlinks:
            continue
        aliases = [a for a in _fetch_aliases(conn, canonical_id) if a != canonical_name]
        related = _fetch_related(conn, canonical_id)

        fname_slug = f"{type_.lower()}-{_slug(canonical_name)}"
        # collision guard (different entities, same slug → disambiguate with id)
        if fname_slug in seen_slugs:
            fname_slug = f"{fname_slug}-{canonical_id}"
        seen_slugs.add(fname_slug)

        frontmatter = {
            "title": canonical_name,
            "kind": "entity",
            "entity_type": type_,
            "canonical_id": canonical_id,
            "aliases": aliases,
            "backlink_count": len(backlinks),
            "relation_count": len(related),
            "updated": date.today().isoformat(),
        }

        lines = [f"# {canonical_name}", ""]
        if aliases:
            lines.append(f"**Also known as:** {', '.join(aliases)}")
            lines.append("")
        lines.append(f"**Type:** `{type_}`")
        lines.append("")
        lines.append(f"## Appears in ({len(backlinks)})")
        lines.append("")
        for pid in backlinks:
            title = _lookup_page_title(Path(wiki_dir), pid)
            # Obsidian-style wiki-link works if the page exists at its slug.
            stem = Path(pid).stem
            lines.append(f"- [[{stem}|{title}]]  \n  `{pid}`")
        lines.append("")

        if related:
            lines.append(f"## Related entities ({len(related)})")
            lines.append("")
            for name, typ, rel_type in related:
                other_slug = f"{typ.lower()}-{_slug(name)}"
                lines.append(f"- **{rel_type}** → [[{other_slug}|{name}]] *(`{typ}`)*")
            lines.append("")

        body = "\n".join(lines)
        path = entities_dir / f"{fname_slug}.md"
        write_page(Page(path=path, frontmatter=frontmatter, body=body))
        written += 1

    log.info(
        "entity pages rebuilt",
        extra={"metadata": {"written": written, "total_canonicals": len(canonicals)}},
    )
    return written
