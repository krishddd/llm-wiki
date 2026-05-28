"""Memory evolution / Reconciler — A-Mem (NeurIPS 2025) Zettelkasten pattern.

After a new source is ingested, the wiki's "compounding" promise requires that
EXISTING pages on overlapping topics get refined to reflect the new evidence —
not just left frozen while a new sibling page is added.

This module provides:
- `find_affected_pages()`  — given the new source's entities, surface pre-existing
   pages most likely affected (entity-overlap + graph 2-hop).
- `propose_edits()`        — for each affected page, ask qwen to propose specific
   edits (insertions / refinements / contradictions to flag) given the new source.
- `apply_edit()`           — apply a single proposed edit to a wiki page,
   appending the original block to a `## Superseded` section so history is preserved.
- `should_auto_apply()`    — gate auto-application on confidence; below threshold
   the proposal is staged for human review under `wiki/review/edits/`.

Bi-temporal hook: when an edit *changes* an existing fact, we close the old
`facts` row (valid_to = today, superseded_by = new fact id) instead of mutating it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..llm import OllamaClient

log = logging.getLogger(__name__)


_RECONCILE_SYSTEM = (
    "You are a careful wiki editor. Given a NEW source and an EXISTING page on a "
    "related topic, decide whether the new source should refine, supplement, or "
    "contradict the existing page. Be conservative — propose changes only when "
    "the new source clearly adds, refines, or contradicts a specific claim.\n\n"
    "Reply ONLY JSON:\n"
    '{"action":"none|append|refine|contradict",'
    ' "edit_kind":"new_section|inline_addition|claim_revision|contradiction_flag|none",'
    ' "title":"<short heading for the new content, if any>",'
    ' "content":"<markdown to insert / replace>",'
    ' "old_text":"<exact substring of EXISTING page to replace, or empty for append>",'
    ' "rationale":"<one short sentence>",'
    ' "confidence":0.XX}\n'
    "Use action='none' if the new source doesn't meaningfully affect this page."
)

_RECONCILE_THRESHOLD_AUTO = 0.80   # auto-apply edits at or above this confidence
_RECONCILE_THRESHOLD_PROPOSE = 0.55  # propose under this threshold = stage in review only


@dataclass
class ReconcileProposal:
    target_page: str               # page_id of existing page being edited
    action: str                    # "none" | "append" | "refine" | "contradict"
    edit_kind: str                 # see schema above
    title: str                     # heading if applicable
    content: str                   # markdown to insert
    old_text: str                  # substring to replace (empty for append)
    rationale: str
    confidence: float
    new_source: str                # page_id or filename of the source that triggered this


@dataclass
class ReconcileBatch:
    proposals: list[ReconcileProposal] = field(default_factory=list)
    applied: list[ReconcileProposal] = field(default_factory=list)
    staged: list[ReconcileProposal] = field(default_factory=list)


def _extract_json(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?\n?", "", (s or "").strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def find_affected_pages(
    *,
    graph,
    new_entities: list[str],
    new_page_id: str,
    max_pages: int = 5,
    min_overlap: int = 2,
) -> list[str]:
    """Return up to `max_pages` existing page_ids that overlap by entity citation
    with the entities extracted from the new source. Excludes the new page itself.

    `min_overlap=2` requires at least 2 shared entities — eliminates spurious
    cross-domain reconciliations driven by a single coincidental entity match
    (e.g. both pages happen to mention "Python" or "USA"). Tune up if you mix
    very disparate domains in one wiki.
    """
    if not graph or not new_entities:
        return []
    candidates: dict[str, int] = {}
    for name in new_entities[:20]:
        try:
            for pid in await graph.pages_for_entity(name, limit=20):
                if pid == new_page_id:
                    continue
                candidates[pid] = candidates.get(pid, 0) + 1
        except Exception as e:
            log.debug("pages_for_entity failed", extra={"metadata": {"error": str(e)[:120]}})
            continue
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, count in ranked[:max_pages] if count >= min_overlap]


async def propose_edit(
    client: OllamaClient,
    *,
    new_source_summary: str,
    new_source_title: str,
    target_page_id: str,
    target_page_body: str,
    target_page_title: str,
) -> ReconcileProposal | None:
    prompt = (
        f"NEW SOURCE: {new_source_title}\n\n"
        f"NEW SOURCE SUMMARY:\n{new_source_summary[:3000]}\n\n"
        f"EXISTING PAGE: {target_page_title}\n\n"
        f"EXISTING PAGE CONTENT:\n{target_page_body[:4000]}\n\n"
        "Should the existing page be edited in light of the new source?"
    )
    try:
        raw = await client.qwen(prompt, system=_RECONCILE_SYSTEM, temperature=0.2)
        d = _extract_json(raw) or {}
    except Exception as e:
        log.debug("propose_edit failed", extra={"metadata": {"error": str(e)[:120]}})
        return None
    action = str(d.get("action", "none")).lower()
    if action not in ("append", "refine", "contradict"):
        return None
    return ReconcileProposal(
        target_page=target_page_id,
        action=action,
        edit_kind=str(d.get("edit_kind", "inline_addition")),
        title=str(d.get("title", "") or "")[:120],
        content=str(d.get("content", "") or "")[:3000],
        old_text=str(d.get("old_text", "") or "")[:1500],
        rationale=str(d.get("rationale", "") or "")[:300],
        confidence=float(d.get("confidence", 0.5) or 0.5),
        new_source=new_source_title,
    )


def should_auto_apply(p: ReconcileProposal) -> bool:
    return p.confidence >= _RECONCILE_THRESHOLD_AUTO and p.action in ("append", "refine")


def should_propose(p: ReconcileProposal) -> bool:
    return p.confidence >= _RECONCILE_THRESHOLD_PROPOSE


def apply_edit_to_body(body: str, proposal: ReconcileProposal) -> tuple[str, bool]:
    """Apply the proposal to a page body. Always preserves history under
    a `## Superseded ...` section so we never lose information.

    Returns (new_body, applied). `applied=False` when nothing was actually
    changed — caller should NOT bump `evolved_by` in that case.
    """
    today = date.today().isoformat()
    header = f"\n\n> [!info] Updated {today} from **{proposal.new_source}**\n> {proposal.rationale}\n"

    if proposal.action == "append" or not proposal.old_text:
        if not proposal.content.strip():
            return body, False
        block = f"\n\n## {proposal.title or 'Update'} (added {today})\n{header}\n{proposal.content.strip()}\n"
        return body.rstrip() + block, True

    if proposal.action == "refine" and proposal.old_text:
        if proposal.old_text not in body:
            log.warning(
                "refine proposal skipped: old_text not found in target body",
                extra={"metadata": {
                    "target": proposal.target_page,
                    "src": proposal.new_source,
                    "old_text_preview": proposal.old_text[:120],
                }},
            )
            return body, False
        new_block = f"{proposal.content.strip()}\n{header}"
        new_body = body.replace(proposal.old_text, new_block, 1)
        new_body += (
            f"\n\n## Superseded ({today}) — from earlier sources\n\n"
            f"> _Replaced by update from {proposal.new_source}_\n\n"
            f"{proposal.old_text.strip()}\n"
        )
        return new_body, True

    if proposal.action == "contradict":
        flag = (
            f"\n\n> [!warning] Contradiction flagged ({today})\n"
            f"> {proposal.new_source}: {proposal.rationale}\n"
            f"> Specifically conflicts with: \"{proposal.old_text[:200]}\"\n"
        )
        return body.rstrip() + flag, True

    return body, False


def stage_proposal(wiki_dir: Path, proposal: ReconcileProposal) -> Path:
    """Write a proposal to wiki/review/edits/<date>-<target>.md for human review."""
    edits_dir = Path(wiki_dir) / "review" / "edits"
    edits_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    safe = re.sub(r"[^A-Za-z0-9_\-]", "-", proposal.target_page)[:80]
    path = edits_dir / f"{today}-{safe}-{proposal.action}.md"
    fm = (
        "---\n"
        f"target_page: {proposal.target_page}\n"
        f"new_source: {proposal.new_source}\n"
        f"action: {proposal.action}\n"
        f"edit_kind: {proposal.edit_kind}\n"
        f"confidence: {proposal.confidence:.2f}\n"
        f"created: {today}\n"
        "---\n\n"
    )
    body = (
        f"# Edit proposal for `{proposal.target_page}`\n\n"
        f"**Trigger:** {proposal.new_source}\n\n"
        f"**Rationale:** {proposal.rationale}\n\n"
        f"## Proposed action: `{proposal.action}` / `{proposal.edit_kind}`\n\n"
    )
    if proposal.title:
        body += f"### {proposal.title}\n\n"
    if proposal.old_text:
        body += "**Replace:**\n\n```\n" + proposal.old_text + "\n```\n\n"
    body += "**With / append:**\n\n" + proposal.content + "\n"
    path.write_text(fm + body, encoding="utf-8")
    return path
