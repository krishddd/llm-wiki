"""Knowledge graph (SQLite + networkx) with fuzzy entity canonicalization."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

log = logging.getLogger(__name__)

ENTITY_TYPES = {"PERSON", "ORG", "CONCEPT", "PLACE", "EVENT"}
RELATION_TYPES = {"RELATES_TO", "PART_OF", "CONTRADICTS", "SUPPORTS", "AUTHORED_BY", "OCCURRED_IN"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    canonical_id INTEGER,
    UNIQUE(name, type)
);
CREATE TABLE IF NOT EXISTS relations (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    rel_type TEXT NOT NULL,
    source_page TEXT,
    -- Bi-temporal columns (Graphiti / Zep pattern):
    -- valid_from = when the fact became true in the world (event time)
    -- valid_to   = when it ceased to be true (NULL = currently believed)
    -- ingested_at= when WE learned it (transaction time)
    valid_from TEXT,
    valid_to   TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    superseded_by INTEGER,  -- id of the relation row that superseded this one (NULL if none)
    PRIMARY KEY (src, dst, rel_type, ingested_at)
);
CREATE TABLE IF NOT EXISTS page_entities (
    page_id TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (page_id, entity_id)
);

-- Bi-temporal facts table — entity-level claims extracted from sources.
-- E.g. ("Active Inference", "addresses", "hallucination") with validity window.
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,           -- entity id (canonical)
    predicate TEXT NOT NULL,               -- short verb-phrase
    object_text TEXT NOT NULL,             -- free-form RHS (entity name OR literal)
    object_id INTEGER,                     -- if RHS is an entity, its canonical id
    source_page TEXT NOT NULL,
    confidence REAL DEFAULT 0.7,
    valid_from TEXT,
    valid_to   TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    superseded_by INTEGER                  -- another facts.id that replaced this fact
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_active  ON facts(subject_id, valid_to);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_canon ON entities(canonical_id);
CREATE INDEX IF NOT EXISTS idx_pe_entity ON page_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_active ON relations(src, dst, valid_to);
"""


def _migrate_relations_table(conn: sqlite3.Connection) -> None:
    """Idempotent ALTER TABLE for older DBs that pre-date the bi-temporal columns."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(relations)")
    cols = {row[1] for row in cur.fetchall()}
    for col, ddl in (
        ("valid_from", "ALTER TABLE relations ADD COLUMN valid_from TEXT"),
        ("valid_to", "ALTER TABLE relations ADD COLUMN valid_to TEXT"),
        ("ingested_at", "ALTER TABLE relations ADD COLUMN ingested_at TEXT"),
        ("superseded_by", "ALTER TABLE relations ADD COLUMN superseded_by INTEGER"),
    ):
        if col not in cols:
            with contextlib.suppress(sqlite3.OperationalError):
                cur.execute(ddl)
    conn.commit()


def _migrate_lifecycle_columns(conn: sqlite3.Connection) -> None:
    """Phase B1: add lifecycle columns (access_count, last_accessed, last_reinforced)
    to facts. Idempotent. Also creates page_access table keyed by page_id."""
    cur = conn.cursor()
    # Facts: add lifecycle columns
    cur.execute("PRAGMA table_info(facts)")
    cols = {row[1] for row in cur.fetchall()}
    for col, ddl in (
        ("access_count", "ALTER TABLE facts ADD COLUMN access_count INTEGER DEFAULT 0"),
        ("last_accessed", "ALTER TABLE facts ADD COLUMN last_accessed TEXT"),
        ("last_reinforced", "ALTER TABLE facts ADD COLUMN last_reinforced TEXT"),
        # original_confidence: snapshot of the value at insert time, never mutated
        # by decay_sweep — prevents compounding decay across multiple sweeps.
        ("original_confidence", "ALTER TABLE facts ADD COLUMN original_confidence REAL"),
    ):
        if col not in cols:
            with contextlib.suppress(sqlite3.OperationalError):
                cur.execute(ddl)
    # Backfill original_confidence for any pre-existing rows.
    with contextlib.suppress(sqlite3.OperationalError):
        cur.execute(
            "UPDATE facts SET original_confidence = confidence "
            "WHERE original_confidence IS NULL"
        )
    # Page-level access counters
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS page_access (
            page_id TEXT PRIMARY KEY,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            last_reinforced TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_page_access_last_accessed ON page_access(last_accessed)")
    conn.commit()


@dataclass
class ExtractedEntity:
    name: str
    type: str


@dataclass
class ExtractedRelation:
    src_name: str
    src_type: str
    dst_name: str
    dst_type: str
    rel_type: str


class KnowledgeGraph:
    def __init__(self, db_path: Path, fuzzy_threshold: int = 95):
        self.db_path = Path(db_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        _migrate_relations_table(self._conn)
        _migrate_lifecycle_columns(self._conn)
        self._conn.commit()
        self._lock = asyncio.Lock()

    def close(self) -> None:
        self._conn.close()

    def _canonicalize(self, name: str, type_: str) -> int:
        """Return the canonical entity id for (name, type). Creates one if no fuzzy match ≥ threshold."""
        from thefuzz import fuzz
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, canonical_id FROM entities WHERE type = ?", (type_,))
        rows = cur.fetchall()
        best_id: int | None = None
        best_score = 0
        for eid, existing_name, canonical_id in rows:
            score = fuzz.token_sort_ratio(name.lower(), existing_name.lower())
            if score > best_score:
                best_score = score
                best_id = canonical_id if canonical_id is not None else eid
        if best_id is not None and best_score >= self.fuzzy_threshold:
            with contextlib.suppress(sqlite3.Error):
                cur.execute(
                    "INSERT OR IGNORE INTO entities(name, type, canonical_id) VALUES (?, ?, ?)",
                    (name, type_, best_id),
                )
            self._conn.commit()
            return best_id
        cur.execute("INSERT OR IGNORE INTO entities(name, type) VALUES (?, ?)", (name, type_))
        cur.execute("SELECT id FROM entities WHERE name=? AND type=?", (name, type_))
        row = cur.fetchone()
        self._conn.commit()
        return int(row[0])

    async def upsert_entities_delta(
        self,
        page_id: str,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation] | None = None,
    ) -> list[int]:
        """Upsert entities + page links + relations. Returns canonical ids linked to the page."""
        relations = relations or []
        async with self._lock:
            canon_ids: list[int] = []
            for ent in entities:
                t = ent.type.upper()
                if t not in ENTITY_TYPES:
                    continue
                eid = self._canonicalize(ent.name.strip(), t)
                self._conn.execute(
                    "INSERT OR IGNORE INTO page_entities(page_id, entity_id) VALUES (?, ?)", (page_id, eid)
                )
                canon_ids.append(eid)
            for rel in relations:
                if rel.rel_type.upper() not in RELATION_TYPES:
                    continue
                src = self._canonicalize(rel.src_name.strip(), rel.src_type.upper())
                dst = self._canonicalize(rel.dst_name.strip(), rel.dst_type.upper())
                rt = rel.rel_type.upper()
                # With bi-temporal PK = (src, dst, rel_type, ingested_at) the naive
                # INSERT OR IGNORE no longer dedupes by triple. Skip insert if an
                # ACTIVE edge (valid_to IS NULL) already exists for this triple.
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT 1 FROM relations WHERE src=? AND dst=? AND rel_type=? "
                    "AND (valid_to IS NULL OR valid_to = '') LIMIT 1",
                    (src, dst, rt),
                )
                if cur.fetchone() is not None:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO relations(src, dst, rel_type, source_page) "
                    "VALUES (?, ?, ?, ?)",
                    (src, dst, rt, page_id),
                )
            self._conn.commit()
            return canon_ids

    async def neighbors_of_pages(self, page_ids: list[str], hops: int = 2) -> list[str]:
        """For the given page_ids, return page_ids that reference entities within `hops` in the graph."""
        if not page_ids:
            return []
        async with self._lock:
            cur = self._conn.cursor()
            placeholders = ",".join(["?"] * len(page_ids))
            cur.execute(f"SELECT DISTINCT entity_id FROM page_entities WHERE page_id IN ({placeholders})", page_ids)
            seed = {row[0] for row in cur.fetchall()}
            frontier = set(seed)
            reachable = set(seed)
            for _ in range(hops):
                if not frontier:
                    break
                qs = ",".join(["?"] * len(frontier))
                cur.execute(f"SELECT dst FROM relations WHERE src IN ({qs})", list(frontier))
                nexts = {row[0] for row in cur.fetchall()}
                cur.execute(f"SELECT src FROM relations WHERE dst IN ({qs})", list(frontier))
                nexts |= {row[0] for row in cur.fetchall()}
                frontier = nexts - reachable
                reachable |= frontier
            if not reachable:
                return []
            qs = ",".join(["?"] * len(reachable))
            cur.execute(
                f"SELECT DISTINCT page_id FROM page_entities WHERE entity_id IN ({qs}) AND page_id NOT IN ({placeholders})",
                list(reachable) + list(page_ids),
            )
            return [row[0] for row in cur.fetchall()]

    async def pages_for_entity(self, name: str, limit: int = 10) -> list[str]:
        """Return page_ids that cite an entity matching `name` (canonical-aware)."""
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, canonical_id FROM entities WHERE LOWER(name) = LOWER(?)", (name,))
            rows = cur.fetchall()
            if not rows:
                return []
            ids: set[int] = set()
            for rid, canon in rows:
                root = canon if canon is not None else rid
                ids.add(int(root))
            if not ids:
                return []
            qs = ",".join(["?"] * len(ids))
            cur.execute(
                f"SELECT DISTINCT pe.page_id FROM page_entities pe "
                f"JOIN entities e ON pe.entity_id = e.id "
                f"WHERE e.id IN ({qs}) OR e.canonical_id IN ({qs}) LIMIT ?",
                list(ids) + list(ids) + [limit],
            )
            return [str(r[0]) for r in cur.fetchall()]

    # ─────────────────────────────────────────────────────────────────
    # Bi-temporal facts API (Graphiti / Zep pattern, 2026)
    # ─────────────────────────────────────────────────────────────────

    async def add_fact(
        self,
        *,
        subject_name: str,
        subject_type: str,
        predicate: str,
        object_text: str,
        object_type: str | None = None,
        source_page: str,
        confidence: float = 0.7,
        valid_from: str | None = None,
    ) -> int:
        """Insert a fact. Subject (and object, if `object_type` given) are canonicalised."""
        async with self._lock:
            subj_id = self._canonicalize(subject_name.strip(), subject_type.upper())
            obj_id: int | None = None
            if object_type:
                obj_id = self._canonicalize(object_text.strip(), object_type.upper())
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO facts(subject_id, predicate, object_text, object_id, "
                "source_page, confidence, original_confidence, valid_from) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (subj_id, predicate.strip(), object_text.strip(), obj_id,
                 source_page, float(confidence), float(confidence), valid_from),
            )
            fid = int(cur.lastrowid)
            self._conn.commit()
            return fid

    async def supersede_fact(self, *, old_fact_id: int, new_fact_id: int, valid_to: str | None = None) -> None:
        """Mark `old_fact_id` as superseded by `new_fact_id` as of `valid_to` (defaults to now)."""
        from datetime import datetime
        ts = valid_to or datetime.now(UTC).date().isoformat()
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ? AND valid_to IS NULL",
                (ts, new_fact_id, old_fact_id),
            )
            self._conn.commit()

    async def active_facts_for(self, subject_name: str) -> list[dict]:
        """Return currently-valid (`valid_to IS NULL`) facts about an entity, canonical-aware."""
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, canonical_id FROM entities WHERE LOWER(name) = LOWER(?)", (subject_name,))
            rows = cur.fetchall()
            if not rows:
                return []
            roots = {int(c) if c is not None else int(rid) for rid, c in rows}
            qs = ",".join(["?"] * len(roots))
            cur.execute(
                f"SELECT id, predicate, object_text, source_page, confidence, "
                f"valid_from, ingested_at, last_reinforced, last_accessed, access_count "
                f"FROM facts WHERE subject_id IN ({qs}) AND valid_to IS NULL "
                f"ORDER BY ingested_at DESC",
                list(roots),
            )
            return [
                {"id": int(r[0]), "predicate": r[1], "object": r[2], "source_page": r[3],
                 "confidence": float(r[4] or 0.0), "valid_from": r[5], "ingested_at": r[6],
                 "last_reinforced": r[7], "last_accessed": r[8],
                 "access_count": int(r[9] or 0)}
                for r in cur.fetchall()
            ]

    async def history_for(self, subject_name: str) -> list[dict]:
        """Return ALL facts (active + superseded) for an entity. Useful for audit + entity pages."""
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, canonical_id FROM entities WHERE LOWER(name) = LOWER(?)", (subject_name,))
            rows = cur.fetchall()
            if not rows:
                return []
            roots = {int(c) if c is not None else int(rid) for rid, c in rows}
            qs = ",".join(["?"] * len(roots))
            cur.execute(
                f"SELECT id, predicate, object_text, source_page, confidence, "
                f"valid_from, valid_to, ingested_at, superseded_by, "
                f"last_reinforced, last_accessed, access_count "
                f"FROM facts WHERE subject_id IN ({qs}) ORDER BY ingested_at",
                list(roots),
            )
            return [
                {"id": int(r[0]), "predicate": r[1], "object": r[2], "source_page": r[3],
                 "confidence": float(r[4] or 0.0), "valid_from": r[5], "valid_to": r[6],
                 "ingested_at": r[7], "superseded_by": r[8],
                 "last_reinforced": r[9], "last_accessed": r[10],
                 "access_count": int(r[11] or 0),
                 "active": r[6] is None}
                for r in cur.fetchall()
            ]

    def stats(self) -> dict:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        n_ent = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM relations")
        n_rel = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT page_id) FROM page_entities")
        n_pages = cur.fetchone()[0]
        return {"entities": n_ent, "relations": n_rel, "pages_with_entities": n_pages}
