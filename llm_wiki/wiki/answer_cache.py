"""Semantic answer cache — embedding-similarity short-circuit for repeat questions.

The procedural store (`wiki/procedures.py`) already recalls answers for questions
whose EXACT normalized pattern hash recurs. This complements it with a *fuzzy* layer:
embed the question and, if a recently-answered question is within `sim_threshold`
cosine, return its stored answer without re-running retrieval + synthesis.

Design notes:
- Default OFF (see `Settings.query_answer_cache`) — serving a cached answer is a
  behaviour change, so it is opt-in.
- Entries carry a TTL and a max-count cap; expired / overflow rows are pruned on
  write so a wiki that has since changed doesn't keep serving stale answers.
- Only sufficiently-confident answers are stored (`min_confidence`), so a weak answer
  is never cached and re-served.
- Stored payload is a JSON-safe subset of `QueryResult`; the query engine rebuilds a
  faithful result from it (blocks are re-parsed from the answer markdown).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    vec TEXT NOT NULL,           -- JSON list[float]
    payload TEXT NOT NULL,       -- JSON answer payload
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL     -- epoch seconds
);
"""


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticAnswerCache:
    """SQLite-backed cosine-similarity cache of recent answers."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> float:
        # Wall clock is fine here — the cache is advisory and its correctness only
        # depends on monotonic-enough ordering + TTL, not on Date-determinism.
        return time.time()

    def lookup(
        self, vec: list[float], *, threshold: float, ttl_days: int
    ) -> dict | None:
        """Return the payload of the most-similar non-expired entry ≥ threshold, else None."""
        if not vec:
            return None
        cutoff = self._now() - ttl_days * 86400
        cur = self._conn.cursor()
        cur.execute(
            "SELECT question, vec, payload FROM answer_cache WHERE created_at >= ? "
            "ORDER BY created_at DESC LIMIT 500",
            (cutoff,),
        )
        best_payload: dict | None = None
        best_sim = threshold
        for question, vec_json, payload_json in cur.fetchall():
            try:
                other = json.loads(vec_json)
            except (json.JSONDecodeError, TypeError):
                continue
            sim = _cosine(vec, other)
            if sim >= best_sim:
                try:
                    best_payload = json.loads(payload_json)
                    best_sim = sim
                    best_payload["_cache_similarity"] = round(sim, 4)
                    best_payload["_cache_question"] = question
                except (json.JSONDecodeError, TypeError):
                    continue
        return best_payload

    def store(
        self, question: str, vec: list[float], payload: dict, confidence: float,
        *, max_entries: int, ttl_days: int,
    ) -> None:
        """Persist an answer and prune expired / overflow rows. Best-effort."""
        if not vec or not question:
            return
        try:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO answer_cache(question, vec, payload, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, json.dumps(vec), json.dumps(payload), float(confidence), self._now()),
            )
            # Prune expired.
            cutoff = self._now() - ttl_days * 86400
            cur.execute("DELETE FROM answer_cache WHERE created_at < ?", (cutoff,))
            # Prune overflow (keep the newest `max_entries`).
            cur.execute(
                "DELETE FROM answer_cache WHERE id NOT IN "
                "(SELECT id FROM answer_cache ORDER BY created_at DESC LIMIT ?)",
                (max_entries,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            log.debug("answer cache store failed", extra={"metadata": {"error": str(e)[:120]}})
