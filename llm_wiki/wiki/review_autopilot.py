"""Review Autopilot — evidence-grounded auto-verification of staged review pages.

Pages that miss the ingest confidence gate land in `wiki/review/` and used to wait
for a human accept/reject. The ingest-time score is a *self-assessment* (the model
rates its own summary without re-reading the source), which is noisy — most staged
pages are fine. This module closes the loop with a verification pass that is
strictly stronger than the original score:

1. **Evidence-grounded judge** — one summary-role call sees BOTH the original
   source excerpt and the staged page, and scores faithfulness (claims supported
   by the source) and coverage (main points captured). Grounded comparison, not
   self-assessment.
2. **Deterministic cross-check** — the fraction of the page's `entity_refs` that
   literally appear in the source text. No LLM, cannot hallucinate.
3. **Composite score** = 0.7 × judge + 0.3 × entity-grounding (judge-only when the
   page has too few entities to ground).
4. **Second opinion** — when the composite lands near a decision boundary, the
   reason role re-judges (a *different* model when roles are split) and the two
   are averaged. Multi-judge agreement suppresses single-call noise.
5. **Three-way decision** (conservative by default):
   - composite ≥ `review_accept_threshold` (0.70) → auto-accept: moved to
     `wiki/sources/`, indexed, audit-logged `WIKI_REVIEW_ACCEPT (auto)`.
   - composite ≤ `review_reject_threshold` (0.30) → auto-archive: moved to
     `wiki/archive/` — never deleted, fully reversible; `WIKI_REVIEW_REJECT (auto)`.
   - in between → stays in review, but the page frontmatter gains an
     `auto_review` block (scores + judge reasons) so the human knows exactly
     what to check.

Pages whose original source can't be re-read (missing file, save-back pages) are
left for the human — the autopilot only acts when it can verify against evidence.

Entry points: called inline at the end of ingest for freshly staged pages
(`REVIEW_AUTOPILOT_ENABLED`), the daily `review_autopilot` scheduler job for the
backlog, and `POST /admin/run/review_autopilot`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..logging_config import audit
from .pages import Page, read_page, write_page

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict verification judge for a knowledge base. Compare the STAGED "
    "PAGE against the ORIGINAL SOURCE excerpt and score:\n"
    '- "faithfulness" (0.0-1.0): are the page\'s factual claims supported by the source?\n'
    '- "coverage" (0.0-1.0): does the page capture the source\'s main points?\n'
    "Judge ONLY against the given source excerpt. Be strict on invented facts, "
    "lenient on phrasing/compression.\n"
    'Reply ONLY JSON: {"faithfulness":0.XX,"coverage":0.XX,"reasons":["…","…"]} '
    "with at most 2 reasons of at most 15 words each."
)

_SOURCE_EXCERPT_CHARS = 5000
_PAGE_EXCERPT_CHARS = 3500
_MIN_GROUNDABLE_ENTITIES = 3
_BOUNDARY_BAND = 0.08  # second opinion when composite is this close to a threshold


@dataclass
class AutoReviewOutcome:
    page: str
    action: str                 # accepted | archived | annotated | skipped
    composite: float | None = None
    judge: dict | None = None
    entity_grounding: float | None = None
    reason: str = ""


@dataclass
class AutoReviewReport:
    accepted: int = 0
    archived: int = 0
    annotated: int = 0
    skipped: int = 0
    outcomes: list[AutoReviewOutcome] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "archived": self.archived,
            "annotated": self.annotated,
            "skipped": self.skipped,
            "outcomes": [o.__dict__ for o in self.outcomes],
        }


_SCORE_RE = re.compile(
    r'"(faithfulness|coverage)"\s*:\s*(0?\.\d+|[01](?:\.0+)?)', re.IGNORECASE
)


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            pass
    # Salvage: small models sometimes truncate mid-"reasons", leaving invalid JSON.
    # The scores come first in the reply, so recover them directly.
    scores = {k.lower(): float(v) for k, v in _SCORE_RE.findall(s or "")}
    if "faithfulness" in scores and "coverage" in scores:
        return {"faithfulness": scores["faithfulness"],
                "coverage": scores["coverage"],
                "reasons": ["(reply truncated — scores salvaged)"]}
    return None


def _load_source_text(source: str) -> str | None:
    """Re-read the original source document. None when it can't serve as evidence
    (missing file, machine provenance like 'query-save-back', loader failure)."""
    if not source or "/" not in source.replace("\\", "/"):
        return None
    p = Path(source)
    if not p.exists() or not p.is_file():
        return None
    try:
        from ..loaders import load_source
        text = load_source(p)
        if not text or not text.strip():
            return None
        if len(text) <= _SOURCE_EXCERPT_CHARS:
            return text
        # Long documents: sample head + middle + tail rather than just the head,
        # so entity-grounding and the judge see evidence from the whole document
        # (a summary legitimately draws from everywhere).
        head = text[: int(_SOURCE_EXCERPT_CHARS * 0.5)]
        mid_at = len(text) // 2
        mid = text[mid_at : mid_at + int(_SOURCE_EXCERPT_CHARS * 0.3)]
        tail = text[-int(_SOURCE_EXCERPT_CHARS * 0.2):]
        return f"{head}\n[…]\n{mid}\n[…]\n{tail}"
    except Exception as e:
        log.debug("autopilot: source unreadable",
                  extra={"metadata": {"source": source, "error": str(e)[:120]}})
        return None


def entity_grounding(source_text: str, entity_refs: list) -> float | None:
    """Fraction of the page's extracted entities literally present in the source.
    Deterministic faithfulness signal. None when too few groundable entities."""
    hay = (source_text or "").lower()
    considered = 0
    hits = 0
    for name in entity_refs or []:
        n = str(name).strip()
        if len(n) < 4:
            continue
        considered += 1
        if n.lower() in hay:
            hits += 1
        if considered >= 20:
            break
    if considered < _MIN_GROUNDABLE_ENTITIES:
        return None
    return hits / considered


async def _judge_once(call, source_excerpt: str, title: str, body: str, *, attempts: int = 2) -> dict | None:
    """One judge verdict. Small models occasionally emit unparseable JSON, so an
    invalid reply gets one retry before giving up (a failed judge → page is left
    for the human, never auto-acted on)."""
    prompt = (
        f"ORIGINAL SOURCE (excerpt):\n{source_excerpt}\n\n"
        f"STAGED PAGE — {title}:\n{body[:_PAGE_EXCERPT_CHARS]}"
    )
    for attempt in range(attempts):
        try:
            raw = await call(prompt, system=JUDGE_SYSTEM, temperature=0.1)
        except Exception as e:
            log.debug("autopilot judge call failed", extra={"metadata": {"error": str(e)[:160]}})
            return None
        data = _extract_json(raw) or {}
        try:
            faith = max(0.0, min(1.0, float(data.get("faithfulness"))))
            cover = max(0.0, min(1.0, float(data.get("coverage"))))
        except (TypeError, ValueError):
            if attempt + 1 < attempts:
                log.debug("autopilot judge reply unparseable — retrying once")
                continue
            return None
        reasons = [str(r)[:200] for r in (data.get("reasons") or [])][:5]
        return {"faithfulness": faith, "coverage": cover, "reasons": reasons}
    return None


def _judge_score(j: dict) -> float:
    # Faithfulness weighs more: an unfaithful page is worse than an incomplete one.
    return 0.6 * j["faithfulness"] + 0.4 * j["coverage"]


def _composite(judge: dict, grounding: float | None) -> float:
    j = _judge_score(judge)
    if grounding is None:
        return j
    return 0.7 * j + 0.3 * grounding


def _near_boundary(score: float, accept_thr: float, reject_thr: float) -> bool:
    return abs(score - accept_thr) < _BOUNDARY_BAND or abs(score - reject_thr) < _BOUNDARY_BAND


async def _accept(
    page: Page, wiki_dir: Path, bm25, dense, composite: float,
    judge: dict, grounding: float | None, graph=None,
) -> str:
    """Promote a staged page to sources/ via the shared promotion helper.

    Records the verdict as an `auto_review` block on the accepted page too (not just
    the gray-zone ones) so the confidence provenance survives on the page itself.
    Returns the new page-id.
    """
    page.frontmatter["auto_review"] = {
        "composite": round(composite, 2),
        "faithfulness": round(judge["faithfulness"], 2),
        "coverage": round(judge["coverage"], 2),
        "entity_grounding": round(grounding, 2) if grounding is not None else None,
        "reasons": judge.get("reasons") or [],
        "verdict": "auto-accepted",
        "date": datetime.now(UTC).date().isoformat(),
    }
    from .review_promote import promote_review_page
    pid = await promote_review_page(
        page, wiki_dir=Path(wiki_dir), bm25=bm25, dense=dense, graph=graph,
        new_confidence=composite,
    )
    audit(log, "WIKI_REVIEW_ACCEPT", str(Path(wiki_dir) / "sources" / page.path.name),
          by="auto", confidence=round(composite, 2))
    return pid


def _archive(page: Page, wiki_dir: Path, composite: float) -> str:
    """Move a failed page to archive/ (reversible — never deleted)."""
    arch_dir = Path(wiki_dir) / "archive"
    arch_dir.mkdir(parents=True, exist_ok=True)
    dst = arch_dir / page.path.name
    old_path = page.path
    page.path = dst
    write_page(page)
    old_path.unlink(missing_ok=True)
    audit(log, "WIKI_REVIEW_REJECT", str(dst), by="auto",
          composite=round(composite, 2), mode="archived")
    return str(dst)


def _annotate(page: Page, judge: dict, grounding: float | None, composite: float) -> None:
    """Record the verdict on the staged page so human review is informed."""
    page.frontmatter["auto_review"] = {
        "composite": round(composite, 2),
        "faithfulness": round(judge["faithfulness"], 2),
        "coverage": round(judge["coverage"], 2),
        "entity_grounding": round(grounding, 2) if grounding is not None else None,
        "reasons": judge.get("reasons") or [],
        "verdict": "needs-human",
        "date": datetime.now(UTC).date().isoformat(),
    }
    write_page(page)


async def autopilot_review(
    *,
    wiki_dir: Path,
    client,
    bm25=None,
    dense=None,
    graph=None,
    settings=None,
    only_page: str | None = None,
    max_pages: int | None = None,
) -> dict:
    """Run the autopilot over `wiki/review/` (or a single page via `only_page`)."""
    from ..config import get_settings
    s = settings or get_settings()
    accept_thr = getattr(s, "review_accept_threshold", 0.70)
    reject_thr = getattr(s, "review_reject_threshold", 0.30)
    second_opinion = getattr(s, "review_second_opinion", True)
    cap = max_pages if max_pages is not None else getattr(s, "review_autopilot_max_pages", 25)

    wiki_dir = Path(wiki_dir)
    review_dir = wiki_dir / "review"
    report = AutoReviewReport()
    if not review_dir.exists():
        return report.as_dict()

    candidates = sorted(review_dir.glob("*.md"))
    if only_page:
        candidates = [p for p in candidates if p.name == only_page]

    for p in candidates[: max(cap, 1)]:
        try:
            page = read_page(p)
        except Exception as e:
            report.skipped += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="skipped", reason=f"unreadable: {str(e)[:80]}"))
            continue
        fm = page.frontmatter or {}

        source_text = _load_source_text(str(fm.get("source") or ""))
        if source_text is None:
            report.skipped += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="skipped", reason="source not re-readable — left for human"))
            continue

        title = str(fm.get("title") or p.stem)
        judge = await _judge_once(client.summarize, source_text, title, page.body)
        if judge is None:
            report.skipped += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="skipped", reason="judge unavailable"))
            continue

        grounding = entity_grounding(source_text, fm.get("entity_refs") or [])
        composite = _composite(judge, grounding)

        # Borderline → second opinion from the reason role, average the judges.
        if second_opinion and _near_boundary(composite, accept_thr, reject_thr):
            second = await _judge_once(client.reason, source_text, title, page.body)
            if second is not None:
                judge = {
                    "faithfulness": (judge["faithfulness"] + second["faithfulness"]) / 2,
                    "coverage": (judge["coverage"] + second["coverage"]) / 2,
                    "reasons": (judge.get("reasons") or []) + (second.get("reasons") or []),
                }
                composite = _composite(judge, grounding)

        if composite >= accept_thr:
            pid = await _accept(page, wiki_dir, bm25, dense, composite, judge, grounding, graph=graph)
            report.accepted += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="accepted", composite=round(composite, 3),
                judge=judge, entity_grounding=grounding, reason=pid))
        elif composite <= reject_thr:
            dst = _archive(page, wiki_dir, composite)
            report.archived += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="archived", composite=round(composite, 3),
                judge=judge, entity_grounding=grounding, reason=dst))
        else:
            _annotate(page, judge, grounding, composite)
            report.annotated += 1
            report.outcomes.append(AutoReviewOutcome(
                page=p.name, action="annotated", composite=round(composite, 3),
                judge=judge, entity_grounding=grounding,
                reason="gray zone — annotated for human review"))

    log.info("review autopilot pass", extra={"metadata": {
        "accepted": report.accepted, "archived": report.archived,
        "annotated": report.annotated, "skipped": report.skipped,
    }})
    return report.as_dict()
