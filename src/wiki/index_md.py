"""Deterministic regen of wiki/index.md — catalogues sources, entities, procedures, review.

OKF v0.1: the index follows the reserved `index.md` convention — sections of markdown
lists in the form `- [Title](url) - description` for progressive disclosure. The root
index declares the bundle's target spec version via `okf_version` frontmatter (the one
case where a reserved file carries frontmatter).
"""
from __future__ import annotations

from pathlib import Path

from .pages import RESERVED_FILENAMES, read_page

OKF_VERSION = "0.1"

_SECTIONS = [
    ("Sources", "sources"),
    ("Entities", "entities"),
    ("Procedures", "procedures"),
    ("Review (staged)", "review"),
]


def rebuild_index(wiki_dir: Path) -> Path:
    wiki_dir = Path(wiki_dir)
    lines = [
        "---",
        f'okf_version: "{OKF_VERSION}"',
        "---",
        "",
        "# Wiki Index",
        "",
        "_Auto-generated — do not edit by hand._",
    ]
    for section, sub in _SECTIONS:
        d = wiki_dir / sub
        if not d.exists():
            continue
        pages = [p for p in sorted(d.glob("*.md")) if p.name.lower() not in RESERVED_FILENAMES]
        if not pages:
            continue
        lines.append(f"\n## {section}\n")
        for p in pages:
            page = read_page(p)
            fm = page.frontmatter or {}
            title = fm.get("title") or p.stem
            rel = p.relative_to(wiki_dir).as_posix()
            desc = str(fm.get("description") or "").strip()
            conf = fm.get("confidence")
            suffix = f" - {desc}" if desc else ""
            if conf is not None:
                suffix += f" (conf {conf})"
            lines.append(f"- [{title}]({rel}){suffix}")
    out = wiki_dir / "index.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
