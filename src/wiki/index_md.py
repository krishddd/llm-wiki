"""Deterministic regen of wiki/index.md — catalogues all sources + entities."""
from __future__ import annotations

from pathlib import Path

from .pages import read_page


def rebuild_index(wiki_dir: Path) -> Path:
    wiki_dir = Path(wiki_dir)
    lines = ["# Wiki Index\n", "_Auto-generated — do not edit by hand._\n"]
    for section, sub in [("Sources", "sources"), ("Entities", "entities"), ("Review (staged)", "review")]:
        d = wiki_dir / sub
        if not d.exists():
            continue
        pages = sorted(d.glob("*.md"))
        if not pages:
            continue
        lines.append(f"\n## {section}\n")
        for p in pages:
            page = read_page(p)
            fm = page.frontmatter or {}
            title = fm.get("title") or p.stem
            conf = fm.get("confidence")
            rel = p.relative_to(wiki_dir).as_posix()
            extra = f" — conf {conf}" if conf is not None else ""
            lines.append(f"- [{title}]({rel}){extra}")
    out = wiki_dir / "index.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
