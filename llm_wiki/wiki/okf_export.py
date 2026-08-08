"""Export the wiki as a clean, shareable OKF v0.1 bundle.

Includes the stable knowledge tiers — `sources/`, `entities/`, `procedures/` —
and regenerates a conformant root `index.md` (okf_version declared) plus the
operation `log.md`. Excludes what a consumer shouldn't see: `review/` (staged,
unaccepted), `episodic/` (transient), `raw/` (source binaries; `resource` URIs
may dangle — OKF tolerates broken links), and `archive/`.

The output directory is a valid standalone bundle: distribute it as a git repo
or archive, or point any OKF consumer (or this project's own
`import_okf_bundle`) at it.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..loaders.okf_loader import validate_bundle
from .index_md import rebuild_index
from .pages import RESERVED_FILENAMES, read_page

log = logging.getLogger(__name__)

_EXPORT_SUBDIRS = ("sources", "entities", "procedures")


def export_okf_bundle(wiki_dir: Path, out_dir: Path, *, clean: bool = True) -> dict:
    """Copy the stable tiers to `out_dir`, regenerate index.md, validate. Returns stats."""
    wiki_dir = Path(wiki_dir)
    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_nonconformant = 0
    for sub in _EXPORT_SUBDIRS:
        src_sub = wiki_dir / sub
        if not src_sub.exists():
            continue
        dst_sub = out_dir / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_sub.glob("*.md")):
            if p.name.lower() in RESERVED_FILENAMES:
                continue
            try:
                page = read_page(p)
            except Exception:
                skipped_nonconformant += 1
                continue
            if not page.frontmatter or not str(page.frontmatter.get("type", "")).strip():
                # OKF conformance requires `type`; unstamped strays don't ship.
                skipped_nonconformant += 1
                continue
            shutil.copy2(p, dst_sub / p.name)
            copied += 1

    # Reserved files: fresh index over the exported tree; carry the log verbatim.
    rebuild_index(out_dir)
    src_log = wiki_dir / "log.md"
    if src_log.exists():
        shutil.copy2(src_log, out_dir / "log.md")

    report = validate_bundle(out_dir)
    stats = {
        "out_dir": str(out_dir),
        "pages_exported": copied,
        "skipped_nonconformant": skipped_nonconformant,
        "conformant": report["conformant"],
        "violations": report["violations"][:20],
    }
    log.info("OKF bundle exported", extra={"metadata": stats})
    return stats
