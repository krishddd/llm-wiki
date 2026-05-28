"""Wiki page read/write with YAML frontmatter, confidence-gated storage, and PageStore used by retrieval."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import Settings, get_settings
from ..logging_config import audit

log = logging.getLogger(__name__)

_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", name.strip().lower()).strip("-")
    return s[:80] or "page"


def page_id_from_path(path: Path, wiki_dir: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(wiki_dir).resolve())).replace("\\", "/")


@dataclass
class Page:
    path: Path
    frontmatter: dict
    body: str

    def render(self) -> str:
        fm = yaml.safe_dump(self.frontmatter, sort_keys=False).strip()
        return f"---\n{fm}\n---\n\n{self.body.strip()}\n"


def read_page(path: Path) -> Page:
    raw = Path(path).read_text(encoding="utf-8")
    m = _FM_RE.match(raw)
    if not m:
        return Page(path=Path(path), frontmatter={}, body=raw)
    fm = yaml.safe_load(m.group(1)) or {}
    return Page(path=Path(path), frontmatter=fm, body=m.group(2))


def write_page(page: Page) -> None:
    page.path.parent.mkdir(parents=True, exist_ok=True)
    page.path.write_text(page.render(), encoding="utf-8")


def stage_or_publish(
    title: str, body: str, frontmatter: dict, *, settings: Settings | None = None
) -> tuple[Path, bool]:
    """Return (page_path, is_live). is_live=False means sent to wiki/review/."""
    s = settings or get_settings()
    conf = float(frontmatter.get("confidence", 0.0))
    is_live = conf >= s.confidence_threshold
    sub = s.wiki_dir / ("sources" if is_live else "review")
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{_slug(title)}.md"
    write_page(Page(path=path, frontmatter=frontmatter, body=body))
    event = "WIKI_WRITE" if is_live else "WIKI_WRITE_STAGED"
    audit(log, event, str(path), confidence=conf, title=title)
    return path, is_live


class PageStore:
    """Thin façade used by retrieval — returns page text + meta by page_id (relative path)."""

    def __init__(self, wiki_dir: Path | None = None):
        self.wiki_dir = Path(wiki_dir or get_settings().wiki_dir)

    async def get_text(self, page_id: str) -> str:
        p = self.wiki_dir / page_id
        if not p.exists():
            return ""
        page = read_page(p)
        return page.body

    async def get_meta(self, page_id: str) -> dict:
        p = self.wiki_dir / page_id
        if not p.exists():
            return {}
        return read_page(p).frontmatter

    def iter_pages(self, subdirs: tuple[str, ...] = ("sources", "entities", "procedures")):
        # Default now includes the procedural memory tier so it's searchable
        # alongside semantic sources/entities. Callers that want to scope a
        # subset (e.g. lint orphan-detection) can pass a narrower tuple.
        for sub in subdirs:
            d = self.wiki_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*.md")):
                yield page_id_from_path(p, self.wiki_dir), read_page(p)
