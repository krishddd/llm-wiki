"""Wiki page read/write with YAML frontmatter, confidence-gated storage, and PageStore used by retrieval.

OKF conformance (Open Knowledge Format v0.1, github.com/GoogleCloudPlatform/knowledge-catalog):
every non-reserved `.md` page carries a `type` frontmatter field (the one field OKF
requires) plus the recommended `timestamp` (ISO 8601, last modification). Both are
stamped centrally in `write_page` so every writer conforms without repeating itself.
`index.md` and `log.md` are OKF reserved filenames — never treated as concept pages.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..config import Settings, get_settings
from ..logging_config import audit

log = logging.getLogger(__name__)

_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)

# OKF reserved filenames — directory listing / update history, not concept documents.
RESERVED_FILENAMES = {"index.md", "log.md"}

# Internal page `kind` → OKF `type` (short human string; OKF types are not registered
# centrally, consumers must tolerate unknown values).
OKF_TYPE_BY_KIND = {
    "source": "Source Document",
    "synthesis": "Synthesis",
    "promoted": "Promoted Synthesis",
    "crystallized": "Session Digest",
    "procedure": "Procedure",
    "entity": "Entity",
    "episodic": "Episodic Log",
    "topic": "Topic Overview",
}


def okf_type_for(kind: str) -> str:
    return OKF_TYPE_BY_KIND.get((kind or "").strip().lower(), "Document")


def derive_description(body: str, max_chars: int = 200) -> str:
    """First prose sentence of `body`, markdown-stripped — OKF's one-line `description`."""
    for line in (body or "").splitlines():
        t = line.strip()
        if not t or t.startswith(("#", "|", "```", "---", "![", "> ")):
            continue
        # strip common inline markdown
        t = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", t)      # [[page|label]] → label
        t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)             # [label](url) → label
        t = re.sub(r"[*_`]{1,3}", "", t)
        t = re.sub(r"^\s*[-*+]\s+", "", t).strip()
        if len(t) < 20:
            continue
        m = re.match(r"(.+?[.!?])(\s|$)", t)
        sent = (m.group(1) if m else t).strip()
        if len(sent) > max_chars:
            # truncate at a word boundary
            sent = sent[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        return sent
    return ""


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
    # OKF stamping — skip reserved filenames (index.md / log.md carry no frontmatter).
    if page.frontmatter and page.path.name.lower() not in RESERVED_FILENAMES:
        if not page.frontmatter.get("type"):
            page.frontmatter["type"] = okf_type_for(str(page.frontmatter.get("kind", "")))
        if not page.frontmatter.get("description"):
            desc = derive_description(page.body)
            if desc:
                page.frontmatter["description"] = desc
        page.frontmatter["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        # Profile / schema contract — enforce AFTER stamping (so auto-filled fields
        # count as present). "warn" logs + audits; "strict" raises ProfileViolation.
        _enforce_profile(page)
    page.path.parent.mkdir(parents=True, exist_ok=True)
    page.path.write_text(page.render(), encoding="utf-8")


def _enforce_profile(page: Page) -> None:
    """Validate the page against the active profile contract. Best-effort loading —
    only the enforcement itself (strict mode) is allowed to raise."""
    try:
        s = get_settings()
        mode = getattr(s, "profile_enforcement", "warn")
        if (mode or "warn").lower() == "off":
            return
        from .profile import enforce, load_profile
        profile = load_profile(getattr(s, "profile_path", "") or None)
    except Exception:
        return  # never let profile *loading* break a write
    from .profile import ProfileViolation
    try:
        result = enforce(page.path.name, page.frontmatter, profile=profile, mode=mode)
    except ProfileViolation:
        audit(log, "PROFILE_VIOLATION", str(page.path), mode="strict",
              errors="; ".join(_pv_errors_from(page, profile)))
        raise
    if not result.ok:
        audit(log, "PROFILE_VIOLATION", str(page.path), mode="warn",
              errors="; ".join(result.errors))
        log.warning("profile contract violation (warn)",
                    extra={"metadata": {"page": page.path.name, "errors": result.errors}})


def _pv_errors_from(page: Page, profile) -> list[str]:
    from .profile import validate_page
    return validate_page(page.frontmatter, profile=profile).errors


def stage_or_publish(
    title: str, body: str, frontmatter: dict, *, settings: Settings | None = None
) -> tuple[Path, bool]:
    """Return (page_path, is_live). is_live=False means sent to wiki/review/.

    Collision guard: two DIFFERENT documents with the same title slug must not
    silently overwrite each other — if the slug is taken by a page from another
    `source`, the new page gets a short hash suffix. Re-ingesting the same
    source still overwrites its own page (intended).
    """
    s = settings or get_settings()
    conf = float(frontmatter.get("confidence", 0.0))
    is_live = conf >= s.confidence_threshold
    sub = s.wiki_dir / ("sources" if is_live else "review")
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{_slug(title)}.md"
    new_source = str(frontmatter.get("source") or "")
    if path.exists() and new_source:
        try:
            existing_source = str(read_page(path).frontmatter.get("source") or "")
        except Exception:
            existing_source = ""
        if existing_source and existing_source != new_source:
            suffix = hashlib.sha1(new_source.encode("utf-8")).hexdigest()[:8]
            path = sub / f"{_slug(title)}-{suffix}.md"
            log.warning(
                "title slug collision — disambiguating",
                extra={"metadata": {"title": title, "existing": existing_source,
                                    "new": new_source, "path": str(path)}},
            )
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
                if p.name.lower() in RESERVED_FILENAMES:
                    continue  # OKF reserved files are not concept pages
                yield page_id_from_path(p, self.wiki_dir), read_page(p)
