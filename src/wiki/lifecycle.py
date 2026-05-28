"""Memory lifecycle: Ebbinghaus decay + access reinforcement.

The wiki's stored confidence is the value the LLM assigned at ingest. The
*effective* confidence at any later moment depends on how recently the page
was reinforced — this module computes that and exposes helpers to mark pages
accessed (which can flip them from decay back to reinforced).

We never overwrite the stored confidence except when a scheduled `decay_sweep`
job is explicitly triggered. Day-to-day reads compute effective confidence
on the fly so the on-disk number stays a faithful "value at ingest" record.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)


_MIN_CONFIDENCE = 0.05  # floor — even very old, never-reinforced pages stay above this


@dataclass
class LifecycleConfig:
    half_life_days: float = 90.0
    reinforcement_threshold: int = 3   # accesses required to reinforce
    reinforcement_window_days: int = 14
    enabled: bool = True


# ───────────────────────────────────────────────────────────────────
# Pure helpers — no DB
# ───────────────────────────────────────────────────────────────────


def _days_since(iso_ts: str | None, now: datetime | None = None) -> float | None:
    if not iso_ts:
        return None
    try:
        # Tolerate both 'YYYY-MM-DD' and full ISO 8601.
        if len(iso_ts) <= 10:
            ts = datetime.fromisoformat(iso_ts).replace(tzinfo=timezone.utc)
        else:
            ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    n = now or datetime.now(timezone.utc)
    return max(0.0, (n - ts).total_seconds() / 86400.0)


def effective_confidence(
    stored: float,
    last_reinforced: str | None,
    *,
    half_life_days: float = 90.0,
    floor: float = _MIN_CONFIDENCE,
    now: datetime | None = None,
) -> float:
    """Ebbinghaus decay. `stored * exp(-Δdays / half_life)` with a floor.

    If `last_reinforced` is None we use the maximum decay (never reinforced).
    """
    if stored is None:
        return floor
    stored = max(0.0, min(1.0, float(stored)))
    if half_life_days <= 0:
        return stored
    delta = _days_since(last_reinforced, now=now)
    if delta is None:
        # Treat unknown timestamp as "1 half-life ago" → halve the value.
        return max(floor, stored * 0.5)
    decayed = stored * math.exp(-delta / half_life_days)
    return max(floor, decayed)


# ───────────────────────────────────────────────────────────────────
# DB-touching helpers
# ───────────────────────────────────────────────────────────────────


async def mark_accessed(graph, page_ids: list[str], *, cfg: LifecycleConfig) -> None:
    """Increment access counters and possibly trigger a reinforcement.

    A page is *reinforced* (= last_reinforced reset to now) when it has been
    accessed `cfg.reinforcement_threshold` times within the last
    `cfg.reinforcement_window_days`.
    """
    if not cfg.enabled or not page_ids:
        return
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    window_start = (now - timedelta(days=cfg.reinforcement_window_days)).isoformat()

    async with graph._lock:
        cur = graph._conn.cursor()
        for pid in page_ids:
            try:
                # Upsert a row in page_access.
                cur.execute(
                    """
                    INSERT INTO page_access(page_id, access_count, last_accessed)
                    VALUES (?, 1, ?)
                    ON CONFLICT(page_id) DO UPDATE SET
                        access_count = access_count + 1,
                        last_accessed = excluded.last_accessed
                    """,
                    (pid, now_iso),
                )
                # Reinforcement: count accesses inside the window.
                # We don't have per-access timestamps, so use a heuristic:
                # if total count >= threshold AND last_reinforced is None or
                # outside the window, reset last_reinforced.
                cur.execute(
                    "SELECT access_count, last_reinforced FROM page_access WHERE page_id = ?",
                    (pid,),
                )
                row = cur.fetchone()
                if row is None:
                    continue
                count, last_reinforced = int(row[0] or 0), row[1]
                needs_reinforce = (
                    count >= cfg.reinforcement_threshold
                    and (last_reinforced is None or last_reinforced < window_start)
                )
                if needs_reinforce:
                    cur.execute(
                        "UPDATE page_access SET last_reinforced = ?, access_count = 0 "
                        "WHERE page_id = ?",
                        (now_iso, pid),
                    )
            except Exception as e:
                log.debug("mark_accessed failed for %s: %s", pid, str(e)[:120])
                continue
        graph._conn.commit()


async def get_page_lifecycle(graph, page_id: str) -> dict | None:
    """Return the lifecycle row for a page (access_count, last_accessed, last_reinforced)."""
    async with graph._lock:
        cur = graph._conn.cursor()
        cur.execute(
            "SELECT access_count, last_accessed, last_reinforced FROM page_access "
            "WHERE page_id = ?",
            (page_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "access_count": int(row[0] or 0),
            "last_accessed": row[1],
            "last_reinforced": row[2],
        }


async def decay_sweep(graph, *, cfg: LifecycleConfig) -> dict:
    """Bulk-update facts.confidence based on each fact's last_reinforced.

    Returns counts: {"facts_processed", "facts_decayed", "min_after", "max_after"}.

    Idempotent: decays from `original_confidence` (snapshot at insert time)
    rather than the current `confidence`, so running the sweep N times in a
    row produces the SAME result. Compounding decay was the previous bug.
    """
    if not cfg.enabled:
        return {"facts_processed": 0, "facts_decayed": 0, "skipped": "lifecycle_disabled"}
    now = datetime.now(timezone.utc)
    processed = 0
    decayed = 0
    confs_after: list[float] = []
    async with graph._lock:
        cur = graph._conn.cursor()
        cur.execute(
            "SELECT id, confidence, original_confidence, last_reinforced, ingested_at "
            "FROM facts WHERE valid_to IS NULL OR valid_to = ''"
        )
        rows = cur.fetchall()
        for fid, conf, original, last_reinforced, ingested_at in rows:
            processed += 1
            seed = last_reinforced or ingested_at
            # If original_confidence is missing (very old row), backfill from
            # the current confidence on the fly so the next sweep is idempotent.
            base = float(original) if original is not None else float(conf or 0.0)
            new_conf = effective_confidence(
                stored=base,
                last_reinforced=seed,
                half_life_days=cfg.half_life_days,
                now=now,
            )
            if abs(new_conf - float(conf or 0.0)) > 0.01:
                if original is None:
                    cur.execute(
                        "UPDATE facts SET confidence = ?, original_confidence = ? WHERE id = ?",
                        (round(new_conf, 4), base, int(fid)),
                    )
                else:
                    cur.execute(
                        "UPDATE facts SET confidence = ? WHERE id = ?",
                        (round(new_conf, 4), int(fid)),
                    )
                decayed += 1
            confs_after.append(new_conf)
        graph._conn.commit()
    return {
        "facts_processed": processed,
        "facts_decayed": decayed,
        "min_after": min(confs_after) if confs_after else None,
        "max_after": max(confs_after) if confs_after else None,
    }
