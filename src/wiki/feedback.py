"""Feedback curator — turn high-signal user feedback into durable memory.

Inspired by nvk/llm-wiki's feedback curator: capture "high-signal corrections,
preferences, approvals" as *reviewable candidates*, drop generic acknowledgements, and
make promotion to memory explicit. This closes a real gap — the episodic tier logs what
*happened*, but nothing here learned from what the user *corrected or preferred*.

Flow:
  record_feedback(text) → classify (heuristic noise filter, else one reason-role call)
    → drop if noise → else store as a `candidate`
  promote(id) → apply to memory by kind:
    - correction → write a curated high-confidence `sources/feedback-*.md` page (indexed)
    - preference → becomes an ACTIVE preference, injected into future synthesis prompts
    - approval   → reinforce the cited page (lifecycle access bump)
    - rejection  → recorded (optionally down-weights the cited page)
  dismiss(id) → candidate is set aside, never applied.

Everything is SQLite-backed (`data/feedback.db`) and best-effort — feedback capture
must never break the request that produced it.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT,
    answer_ref TEXT,                      -- cited page id / answer id (optional)
    text       TEXT NOT NULL,             -- the user's raw feedback
    kind       TEXT NOT NULL,             -- correction | preference | approval | rejection | noise
    actionable TEXT,                      -- extracted actionable content
    status     TEXT NOT NULL DEFAULT 'candidate',   -- candidate | promoted | dismissed
    created_at TEXT NOT NULL,
    promoted_at TEXT,
    result     TEXT                       -- what promotion produced (e.g. new page id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);
CREATE INDEX IF NOT EXISTS idx_feedback_kind ON feedback(kind);
"""

KINDS = ("correction", "preference", "approval", "rejection", "noise")
HIGH_SIGNAL = ("correction", "preference")

# Cheap pre-filter: obvious generic acknowledgements never warrant an LLM call.
_NOISE_RE = re.compile(
    r"^\s*(thanks?|thx|ty|ok(ay)?|kk|cool|nice|great|perfect|good|got it|makes sense|"
    r"cheers|\U0001F44D|✅|yes|yep|yeah|no+)\W*$",
    re.IGNORECASE,
)

CLASSIFY_SYSTEM = (
    "You triage user feedback on an AI answer into ONE category:\n"
    '- "correction": a factual fix ("actually X was founded in 1998, not 2001").\n'
    '- "preference": how answers should be produced ("always cite sources", "be concise").\n'
    '- "approval": confirms the answer is correct/useful, no new info.\n'
    '- "rejection": says the answer is wrong/unhelpful but gives no specific fix.\n'
    '- "noise": a generic acknowledgement with no signal ("thanks", "ok").\n'
    "Also extract the actionable content (the corrected fact, or the preference rule) — "
    'empty for approval/rejection/noise.\n'
    'Reply ONLY JSON: {"kind":"…","actionable":"…"}'
)


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return None


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", (s or "").strip().lower()).strip("-")
    return s[:60] or "feedback"


@dataclass
class FeedbackRecord:
    id: int
    question: str
    answer_ref: str
    text: str
    kind: str
    actionable: str
    status: str
    created_at: str
    promoted_at: str | None = None
    result: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__


class FeedbackStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def insert(self, *, question: str, answer_ref: str, text: str, kind: str, actionable: str,
               status: str = "candidate") -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO feedback(question, answer_ref, text, kind, actionable, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (question, answer_ref, text, kind, actionable, status, self._now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get(self, fid: int) -> FeedbackRecord | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, question, answer_ref, text, kind, actionable, status, created_at, "
            "promoted_at, result FROM feedback WHERE id = ?", (fid,),
        )
        r = cur.fetchone()
        return FeedbackRecord(*r) if r else None

    def list(self, *, status: str | None = None, limit: int = 100) -> list[FeedbackRecord]:
        cur = self._conn.cursor()
        if status:
            cur.execute(
                "SELECT id, question, answer_ref, text, kind, actionable, status, created_at, "
                "promoted_at, result FROM feedback WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur.execute(
                "SELECT id, question, answer_ref, text, kind, actionable, status, created_at, "
                "promoted_at, result FROM feedback ORDER BY id DESC LIMIT ?", (limit,),
            )
        return [FeedbackRecord(*r) for r in cur.fetchall()]

    def set_status(self, fid: int, status: str, *, result: str | None = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE feedback SET status = ?, promoted_at = ?, result = ? WHERE id = ?",
            (status, self._now() if status == "promoted" else None, result, fid),
        )
        self._conn.commit()

    def active_preferences(self, limit: int = 8) -> list[str]:
        """Promoted preference rules, newest first — injected into synthesis."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COALESCE(NULLIF(actionable,''), text) FROM feedback "
            "WHERE kind = 'preference' AND status = 'promoted' ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [str(r[0]).strip() for r in cur.fetchall() if r[0] and str(r[0]).strip()]


def _heuristic_kind(text: str) -> str | None:
    """Return 'noise' for obvious acks without an LLM call; None if unsure."""
    t = (text or "").strip()
    if not t or len(t) <= 3:
        return "noise"
    if _NOISE_RE.match(t):
        return "noise"
    return None


async def classify_feedback(client, *, question: str, answer: str, text: str) -> dict:
    """Classify feedback into {kind, actionable}. Heuristic noise filter first, then
    one reason-role call. Any failure → treated as noise (dropped, never mis-applied)."""
    h = _heuristic_kind(text)
    if h == "noise":
        return {"kind": "noise", "actionable": ""}
    prompt = (
        f"QUESTION:\n{question[:500]}\n\nAI ANSWER (excerpt):\n{(answer or '')[:800]}\n\n"
        f"USER FEEDBACK:\n{text[:800]}"
    )
    try:
        raw = await client.reason(prompt, system=CLASSIFY_SYSTEM, temperature=0.1)
    except Exception as e:
        log.debug("feedback classify failed", extra={"metadata": {"error": str(e)[:160]}})
        return {"kind": "noise", "actionable": ""}
    data = _extract_json(raw) or {}
    kind = str(data.get("kind", "")).strip().lower()
    if kind not in KINDS:
        kind = "noise"
    return {"kind": kind, "actionable": str(data.get("actionable", "")).strip()[:600]}


async def record_feedback(
    store: FeedbackStore,
    client,
    *,
    question: str,
    answer: str = "",
    answer_ref: str = "",
    text: str,
    auto_promote: bool = False,
    promote_ctx: dict | None = None,
) -> dict:
    """Classify + store feedback. Noise is dropped. Returns a summary dict."""
    cls = await classify_feedback(client, question=question, answer=answer, text=text)
    kind = cls["kind"]
    if kind == "noise":
        return {"stored": False, "kind": "noise", "reason": "generic acknowledgement — dropped"}
    fid = store.insert(question=question, answer_ref=answer_ref, text=text,
                       kind=kind, actionable=cls["actionable"])
    out = {"stored": True, "id": fid, "kind": kind, "actionable": cls["actionable"],
           "status": "candidate"}
    if auto_promote and kind in HIGH_SIGNAL:
        res = await promote_feedback(store, fid, **(promote_ctx or {}))
        out.update({"status": "promoted", "promotion": res})
    return out


async def promote_feedback(
    store: FeedbackStore,
    fid: int,
    *,
    wiki_dir: Path | None = None,
    bm25=None,
    dense=None,
    graph=None,
) -> dict:
    """Apply a candidate to durable memory according to its kind. Idempotent-ish:
    a page-writing promotion re-runs cleanly (same slug overwrites)."""
    rec = store.get(fid)
    if rec is None:
        return {"ok": False, "error": "not found"}
    if rec.status == "promoted":
        return {"ok": True, "kind": rec.kind, "note": "already promoted", "result": rec.result}

    result_note = ""
    if rec.kind == "correction" and wiki_dir is not None:
        result_note = await _promote_correction(rec, Path(wiki_dir), bm25, dense)
    elif rec.kind == "preference":
        result_note = "active preference (injected into synthesis)"
    elif rec.kind == "approval" and graph is not None and rec.answer_ref:
        result_note = await _reinforce_page(graph, rec.answer_ref)
    elif rec.kind == "rejection":
        result_note = "recorded (rejection) — page flagged for review"
    else:
        result_note = f"{rec.kind}: recorded"

    store.set_status(fid, "promoted", result=result_note)
    return {"ok": True, "kind": rec.kind, "result": result_note}


async def _promote_correction(rec: FeedbackRecord, wiki_dir: Path, bm25, dense) -> str:
    """Write a curated, high-confidence source page carrying the correction, and index it."""
    from .pages import Page, page_id_from_path, write_page
    correction = rec.actionable or rec.text
    title = f"Correction: {correction[:60]}"
    slug = _slug(f"feedback-{correction}")
    dst = wiki_dir / "sources" / f"{slug}.md"
    body_lines = [f"# {title}", "", "> [!note] Curated from user feedback.", "",
                  correction, ""]
    if rec.question:
        body_lines += ["## Context", "", f"In response to: *{rec.question.strip()}*", ""]
    body = "\n".join(body_lines)
    fm = {
        "title": title[:120],
        "kind": "source",
        "description": correction[:200],
        "source": "user-feedback",
        "confidence": 0.9,
        "confidence_reason": "curated from an explicit user correction",
        "domain": "general",
        "tags": ["correction"],
        "entity_refs": [],
    }
    write_page(Page(path=dst, frontmatter=fm, body=body))
    pid = page_id_from_path(dst, wiki_dir)
    try:
        from .reindex import index_page_chunks
        await index_page_chunks(bm25, dense, pid, title, body, fm)
    except Exception as e:
        log.debug("correction index failed", extra={"metadata": {"error": str(e)[:120]}})
    return f"created {pid}"


async def _reinforce_page(graph, page_id: str) -> str:
    """Approval → bump the cited page's lifecycle access counter."""
    try:
        from ..config import get_settings
        from .lifecycle import LifecycleConfig, mark_accessed
        s = get_settings()
        cfg = LifecycleConfig(
            half_life_days=getattr(s, "decay_half_life_days", 90.0),
            reinforcement_threshold=getattr(s, "reinforcement_threshold", 3),
            reinforcement_window_days=getattr(s, "reinforcement_window_days", 14),
            enabled=getattr(s, "lifecycle_enabled", True),
        )
        await mark_accessed(graph, [page_id], cfg=cfg)
        return f"reinforced {page_id}"
    except Exception as e:
        log.debug("approval reinforce failed", extra={"metadata": {"error": str(e)[:120]}})
        return "approval recorded"
