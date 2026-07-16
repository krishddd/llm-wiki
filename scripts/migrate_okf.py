"""One-shot migration: re-stamp existing wiki pages to OKF v0.1 conformance.

Adds the missing frontmatter fields to every concept page under wiki/sources,
wiki/entities, wiki/procedures, wiki/review and wiki/episodic:

- `kind`  — inferred from directory / slug prefix when absent (older source pages
            were written without one).
- `type`  — OKF's single required field, mapped from `kind` (entity pages use
            their `entity_type`).
- `description` — first prose sentence of the body when absent.
- `resource`    — mirrored from `source` when it points at a raw file.
- `timestamp`   — stamped by `write_page` on rewrite.

Idempotent: pages already carrying a field are left as-is (timestamp always
refreshes — the migration is itself a modification). Finishes by rebuilding
wiki/index.md in the OKF list format.

Usage:  python -m scripts.migrate_okf [--wiki-dir wiki] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wiki.index_md import rebuild_index                      # noqa: E402
from src.wiki.pages import (                                     # noqa: E402
    RESERVED_FILENAMES,
    derive_description,
    okf_type_for,
    read_page,
    write_page,
)

_OKF_ENTITY_TYPE = {
    "PERSON": "Person",
    "ORG": "Organization",
    "CONCEPT": "Concept",
    "PLACE": "Place",
    "EVENT": "Event",
}

# slug prefix → kind, for pages written before `kind` was stamped consistently
_PREFIX_KINDS = ("synthesis", "promoted", "crystallized", "procedure", "auto")
_DIR_DEFAULT_KIND = {
    "sources": "source",
    "review": "source",
    "entities": "entity",
    "procedures": "procedure",
    "episodic": "episodic",
}


def _infer_kind(page_path: Path, subdir: str, fm: dict) -> str:
    if fm.get("kind"):
        return str(fm["kind"])
    stem = page_path.stem.lower()
    for prefix in _PREFIX_KINDS:
        if stem.startswith(prefix + "-"):
            return "promoted" if prefix == "auto" else prefix
    return _DIR_DEFAULT_KIND.get(subdir, "source")


def migrate(wiki_dir: Path, *, dry_run: bool = False) -> dict:
    stats = {"scanned": 0, "updated": 0, "skipped_no_frontmatter": 0}
    for subdir in ("sources", "entities", "procedures", "review", "episodic"):
        d = wiki_dir / subdir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            if p.name.lower() in RESERVED_FILENAMES:
                continue
            stats["scanned"] += 1
            page = read_page(p)
            fm = page.frontmatter
            if not fm:
                # A bodiless/frontmatterless file — don't invent metadata for it.
                stats["skipped_no_frontmatter"] += 1
                continue

            changed = False
            kind = _infer_kind(p, subdir, fm)
            if fm.get("kind") != kind:
                fm["kind"] = kind
                changed = True
            if not fm.get("type"):
                if kind == "entity" and fm.get("entity_type"):
                    fm["type"] = _OKF_ENTITY_TYPE.get(str(fm["entity_type"]).upper(), "Concept")
                else:
                    fm["type"] = okf_type_for(kind)
                changed = True
            if not fm.get("description"):
                if kind == "entity":
                    # Deterministic formula matching entity_pages.py — the body's first
                    # prose line is just a backlink title, useless as a description.
                    desc = (
                        f"Auto-generated entity page for {fm.get('title', p.stem)} "
                        f"({fm.get('entity_type', 'CONCEPT')}); appears in "
                        f"{fm.get('backlink_count', 0)} wiki page(s)."
                    )
                else:
                    desc = derive_description(page.body)
                if desc:
                    fm["description"] = desc
                    changed = True
            src = str(fm.get("source") or "")
            if not fm.get("resource") and src and ("/" in src or "\\" in src):
                fm["resource"] = src.replace("\\", "/")
                changed = True

            if changed and not dry_run:
                write_page(page)
            if changed:
                stats["updated"] += 1
                print(f"  {'[dry-run] ' if dry_run else ''}updated {p.relative_to(wiki_dir)}"
                      f"  (kind={fm['kind']}, type={fm['type']})")
    if not dry_run:
        rebuild_index(wiki_dir)
        print("index.md rebuilt (OKF list format, okf_version declared)")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stats = migrate(args.wiki_dir.resolve(), dry_run=args.dry_run)
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
