"""Procedural memory tier — repeated query patterns become first-class wiki pages.

When a user asks the same question (or a structurally similar variant) ≥ N times
and consistently retrieves the same set of pages, that's a *procedure* — a
canonical workflow worth crystallising.

We track patterns in a small SQLite table (`procedures` in `data/procedures.db`)
and emit Markdown pages to `wiki/procedures/<slug>.md` once a pattern crosses
the hit threshold.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..llm import OllamaClient
from .pages import Page, write_page

log = logging.getLogger(__name__)


_PROCEDURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    pattern_hash TEXT PRIMARY KEY,
    query_template TEXT NOT NULL,
    intent TEXT,
    canonical_pages TEXT,        -- JSON-encoded list of page_ids
    hit_count INTEGER DEFAULT 1,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    promoted_at TEXT             -- when written to wiki/procedures/
);
CREATE INDEX IF NOT EXISTS idx_procedures_hits ON procedures(hit_count);
CREATE INDEX IF NOT EXISTS idx_procedures_last ON procedures(last_seen);
"""

_PROCEDURE_SYSTEM = (
    "You are documenting a recurring query pattern as a reusable procedure. "
    "Given the canonical query and its retrieval set, produce: a short title (one line), "
    "a one-sentence purpose, a numbered step-by-step procedure (3-7 steps), and a list of "
    "the source pages that anchor each step. "
    "Reply ONLY JSON: "
    '{"title":"…","purpose":"…","steps":["1. …","2. …"],"sources":["page-id"]}'
)


def _normalize(query: str) -> str:
    """Lowercase, strip stopwords/numbers, sort tokens. Used to bucket near-duplicate queries."""
    q = query.lower()
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    toks = sorted(set(t for t in q.split() if len(t) > 3 and t not in _STOP))
    return " ".join(toks)


_STOP = {
    "what", "which", "this", "that", "with", "from", "have", "when", "where",
    "their", "would", "could", "should", "about", "there", "these", "those",
    "into", "than", "then", "such", "your", "more", "tell", "explain", "show",
}


def _pattern_hash(query: str) -> str:
    return hashlib.sha1(_normalize(query).encode("utf-8")).hexdigest()[:16]


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s.strip().lower()).strip("-")
    return s[:80] or "procedure"


# ───────────────────────────────────────────────────────────────────
# Storage
# ───────────────────────────────────────────────────────────────────


@dataclass
class ProcedureStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_PROCEDURES_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record(self, query: str, intent: str | None, top_pages: list[str]) -> tuple[str, int]:
        """Record a query observation. Returns (pattern_hash, new_hit_count)."""
        import json
        ph = _pattern_hash(query)
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT hit_count, canonical_pages FROM procedures WHERE pattern_hash = ?",
            (ph,),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """INSERT INTO procedures(pattern_hash, query_template, intent, canonical_pages,
                                          hit_count, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (ph, query[:300], intent or "synthesis",
                 json.dumps(top_pages[:10]), now, now),
            )
            self._conn.commit()
            return ph, 1

        # Update: bump hit_count, merge page set (last 10), update last_seen.
        prior_pages = []
        try:
            prior_pages = json.loads(row[1] or "[]")
        except Exception:
            prior_pages = []
        merged = list(dict.fromkeys(top_pages + prior_pages))[:10]
        new_count = int(row[0] or 0) + 1
        cur.execute(
            "UPDATE procedures SET hit_count = ?, canonical_pages = ?, last_seen = ? "
            "WHERE pattern_hash = ?",
            (new_count, json.dumps(merged), now, ph),
        )
        self._conn.commit()
        return ph, new_count

    def candidates_for_promotion(self, *, min_hits: int = 5) -> list[dict]:
        """Patterns at or above the threshold that haven't yet been promoted (or whose
        hit_count has grown since promotion)."""
        import json
        cur = self._conn.cursor()
        cur.execute(
            """SELECT pattern_hash, query_template, intent, canonical_pages, hit_count,
                      first_seen, last_seen, promoted_at
               FROM procedures
               WHERE hit_count >= ?
               ORDER BY hit_count DESC""",
            (min_hits,),
        )
        out = []
        for r in cur.fetchall():
            try:
                pages = json.loads(r[3] or "[]")
            except Exception:
                pages = []
            out.append({
                "pattern_hash": r[0], "query_template": r[1], "intent": r[2],
                "canonical_pages": pages, "hit_count": int(r[4] or 0),
                "first_seen": r[5], "last_seen": r[6], "promoted_at": r[7],
            })
        return out

    def mark_promoted(self, pattern_hash: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE procedures SET promoted_at = ? WHERE pattern_hash = ?",
            (datetime.now(timezone.utc).isoformat(), pattern_hash),
        )
        self._conn.commit()


# ───────────────────────────────────────────────────────────────────
# High-level API
# ───────────────────────────────────────────────────────────────────


async def record_query_pattern(
    store: ProcedureStore,
    query: str,
    intent: str | None,
    top_pages: list[str],
) -> tuple[str, int]:
    """Wrapper to be called from the query path."""
    try:
        return store.record(query, intent, top_pages)
    except Exception as e:
        log.debug("record_query_pattern failed",
                  extra={"metadata": {"error": str(e)[:120]}})
        return "", 0


def _extract_json(s: str) -> dict | None:
    import json
    s = re.sub(r"^```(?:json)?\n?", "", (s or "").strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def detect_procedures(
    *,
    store: ProcedureStore,
    wiki_dir: Path,
    client: OllamaClient,
    min_hits: int = 5,
    max_to_promote: int = 5,
) -> dict:
    """Promote qualifying patterns to wiki/procedures/<slug>.md pages.

    Returns counts: {candidates, promoted, pages}.
    """
    candidates = store.candidates_for_promotion(min_hits=min_hits)
    proc_dir = Path(wiki_dir) / "procedures"
    proc_dir.mkdir(parents=True, exist_ok=True)

    promoted: list[str] = []
    for c in candidates[:max_to_promote]:
        if c.get("promoted_at"):
            continue  # already a wiki page; skip until policy says re-emit
        prompt = (
            f"QUERY (recurring {c['hit_count']}x): {c['query_template']}\n"
            f"INTENT: {c['intent']}\n"
            f"RETRIEVAL SET: {', '.join(c['canonical_pages'])}\n\n"
            "Document this as a procedure."
        )
        try:
            raw = await client.qwen(prompt, system=_PROCEDURE_SYSTEM, temperature=0.2)
            parsed = _extract_json(raw) or {}
        except Exception as e:
            log.debug("procedure synth failed", extra={"metadata": {"error": str(e)[:160]}})
            continue
        title = str(parsed.get("title") or c["query_template"])[:120]
        purpose = str(parsed.get("purpose") or "")
        steps = parsed.get("steps") or []
        sources = parsed.get("sources") or c["canonical_pages"]

        body_lines = [f"# {title}", ""]
        if purpose:
            body_lines.append(f"**Purpose:** {purpose}")
            body_lines.append("")
        body_lines.append("## Steps")
        body_lines.append("")
        for s in steps:
            body_lines.append(f"- {s}")
        body_lines.append("")
        body_lines.append("## Anchor pages")
        body_lines.append("")
        for src in sources:
            stem = Path(str(src)).stem
            body_lines.append(f"- [[{stem}]]")
        body = "\n".join(body_lines)

        fm = {
            "title": title,
            "kind": "procedure",
            "purpose": purpose[:400],
            "hit_count": c["hit_count"],
            "first_seen": c["first_seen"],
            "last_seen": c["last_seen"],
            "intent": c["intent"],
            "anchor_pages": list(sources)[:10],
            "pattern_hash": c["pattern_hash"],
            "created": datetime.now(timezone.utc).date().isoformat(),
        }
        path = proc_dir / f"{_slug(title)}.md"
        try:
            write_page(Page(path=path, frontmatter=fm, body=body))
            store.mark_promoted(c["pattern_hash"])
            promoted.append(str(path).replace("\\", "/"))
        except Exception as e:
            log.warning("write procedure page failed",
                        extra={"metadata": {"error": str(e)[:200]}})
            continue

    log.info(
        "procedural promotion",
        extra={"metadata": {"candidates": len(candidates), "promoted": len(promoted)}},
    )
    return {"candidates": len(candidates), "promoted": len(promoted), "pages": promoted}
