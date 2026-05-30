"""Auto-resolve contradictions between two facts.

Composite score per claim:
   0.5 × source_page_confidence
 + 0.3 × recency_score (newer wins)
 + 0.2 × supporting_count (how many other facts on the same subject agree)

If the winning margin ≥ 0.2, supersede the loser. Otherwise, keep both flagged
for human review.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_RESOLVE_MARGIN = 0.2


@dataclass
class FactSnapshot:
    fact_id: int
    subject_id: int
    predicate: str
    object_text: str
    source_page: str
    confidence: float
    ingested_at: str | None


def _read_page_confidence(wiki_dir: Path, page_id: str) -> float:
    """Read the source page's frontmatter confidence (default 0.5 if missing)."""
    try:
        path = Path(wiki_dir) / page_id
        if not path.exists():
            return 0.5
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return 0.5
        end = text.find("\n---", 3)
        if end < 0:
            return 0.5
        fm = yaml.safe_load(text[3:end]) or {}
        return float(fm.get("confidence", 0.5))
    except Exception:
        return 0.5


def _recency_score(ingested_at: str | None, now: datetime | None = None) -> float:
    """Newer = higher (1.0 = today, decays over a year)."""
    if not ingested_at:
        return 0.5
    try:
        if len(ingested_at) <= 10:
            ts = datetime.fromisoformat(ingested_at).replace(tzinfo=UTC)
        else:
            ts = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
    except Exception:
        return 0.5
    n = now or datetime.now(UTC)
    days = max(0.0, (n - ts).total_seconds() / 86400.0)
    # 1.0 today, ~0.5 at 180 days, ~0.0 at 365.
    return max(0.0, 1.0 - (days / 365.0))


async def _supporting_count(graph, subject_id: int, predicate: str, exclude_ids: set[int]) -> int:
    """Count active facts with the same subject+predicate (loose semantic agreement)."""
    async with graph._lock:
        cur = graph._conn.cursor()
        cur.execute(
            "SELECT id FROM facts WHERE subject_id = ? AND predicate = ? "
            "AND (valid_to IS NULL OR valid_to = '')",
            (subject_id, predicate),
        )
        ids = {int(r[0]) for r in cur.fetchall()}
    return len(ids - exclude_ids)


async def _score_fact(graph, wiki_dir: Path, snap: FactSnapshot, exclude_ids: set[int]) -> float:
    page_conf = _read_page_confidence(wiki_dir, snap.source_page)
    recency = _recency_score(snap.ingested_at)
    support = await _supporting_count(graph, snap.subject_id, snap.predicate, exclude_ids)
    # Normalise support: cap at 5 = 1.0
    support_norm = min(1.0, support / 5.0)
    score = 0.5 * page_conf + 0.3 * recency + 0.2 * support_norm
    log.debug(
        "contradiction score",
        extra={"metadata": {
            "fact": snap.fact_id, "page_conf": round(page_conf, 3),
            "recency": round(recency, 3), "support": support, "score": round(score, 3),
        }},
    )
    return score


async def _load_snapshot(graph, fact_id: int) -> FactSnapshot | None:
    async with graph._lock:
        cur = graph._conn.cursor()
        cur.execute(
            "SELECT id, subject_id, predicate, object_text, source_page, confidence, ingested_at "
            "FROM facts WHERE id = ?",
            (fact_id,),
        )
        r = cur.fetchone()
    if r is None:
        return None
    return FactSnapshot(
        fact_id=int(r[0]), subject_id=int(r[1]),
        predicate=str(r[2] or ""), object_text=str(r[3] or ""),
        source_page=str(r[4] or ""),
        confidence=float(r[5] or 0.0), ingested_at=r[6],
    )


async def resolve_contradiction(
    graph,
    fact_a_id: int,
    fact_b_id: int,
    *,
    wiki_dir: Path,
    margin: float = _RESOLVE_MARGIN,
) -> dict:
    """Score both facts; supersede the loser if the margin is decisive.

    Returns:
      {
        "resolved": bool,
        "winner": fact_id | None,
        "loser": fact_id | None,
        "margin": float,
        "reason": str,
      }
    """
    a = await _load_snapshot(graph, fact_a_id)
    b = await _load_snapshot(graph, fact_b_id)
    if a is None or b is None:
        return {"resolved": False, "winner": None, "loser": None,
                "margin": 0.0, "reason": "fact-missing"}
    excl = {fact_a_id, fact_b_id}
    sa = await _score_fact(graph, wiki_dir, a, excl)
    sb = await _score_fact(graph, wiki_dir, b, excl)
    diff = sa - sb
    if abs(diff) < margin:
        return {"resolved": False, "winner": None, "loser": None,
                "margin": round(abs(diff), 3),
                "reason": f"margin {abs(diff):.3f} < {margin:.2f}"}
    if diff > 0:
        winner, loser = a, b
    else:
        winner, loser = b, a
    today = datetime.now(UTC).date().isoformat()
    try:
        await graph.supersede_fact(
            old_fact_id=loser.fact_id,
            new_fact_id=winner.fact_id,
            valid_to=today,
        )
    except Exception as e:
        return {"resolved": False, "winner": winner.fact_id, "loser": loser.fact_id,
                "margin": round(abs(diff), 3),
                "reason": f"supersede_fact raised: {str(e)[:160]}"}
    return {
        "resolved": True,
        "winner": winner.fact_id,
        "loser": loser.fact_id,
        "margin": round(abs(diff), 3),
        "reason": "auto-resolved by composite score",
    }


async def list_unresolved_contradictions(graph, *, limit: int = 50) -> list[dict]:
    """Return facts that contradict an active sibling but haven't been resolved.

    Heuristic: pairs of active facts with same subject_id + predicate but
    different object_text. Returns [{fact_a, fact_b, subject_id, predicate}].
    """
    async with graph._lock:
        cur = graph._conn.cursor()
        cur.execute(
            """
            SELECT a.id, b.id, a.subject_id, a.predicate, a.object_text, b.object_text
            FROM facts a
            JOIN facts b ON a.subject_id = b.subject_id
                         AND a.predicate = b.predicate
                         AND a.id < b.id
                         AND a.object_text <> b.object_text
            WHERE (a.valid_to IS NULL OR a.valid_to = '')
              AND (b.valid_to IS NULL OR b.valid_to = '')
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {"fact_a": int(r[0]), "fact_b": int(r[1]),
         "subject_id": int(r[2]), "predicate": str(r[3]),
         "object_a": str(r[4]), "object_b": str(r[5])}
        for r in rows
    ]
