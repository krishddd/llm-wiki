"""Post-synthesis claim verification (NLI-lite, one batched judge call).

The self-grounding check in `query.py` only verifies that each `[citation]` names a
retrieved page — it can't catch the classic RAG failure of citing the *right page for
the wrong claim*. This module extracts the sentence behind every `[Page]^0.NN` marker,
pairs it with the cited page's snippet, and asks the fast model — in ONE batched call —
whether each claim is supported / partially supported / unsupported by its source.
Verdicts recalibrate the per-claim confidences before they are aggregated.

Best-effort: any failure leaves the original confidences untouched.
"""
from __future__ import annotations

import json
import logging
import re

from .claims import Claim

log = logging.getLogger(__name__)

_VERIFY_SYSTEM = (
    "You are a strict fact-checking judge. For each numbered CLAIM, decide whether "
    "its SOURCE excerpt supports it:\n"
    '- "supported": the excerpt clearly states or directly entails the claim\n'
    '- "partial": the excerpt is related but only partly backs the claim\n'
    '- "unsupported": the excerpt does not back the claim\n'
    'Reply ONLY JSON: {"verdicts":[{"id":1,"verdict":"supported"},…]} — one entry per claim.'
)

# Confidence multipliers per verdict. "unknown" (judge failed / claim unmatched) = no-op.
VERDICT_WEIGHT = {"supported": 1.0, "partial": 0.75, "unsupported": 0.35, "unknown": 1.0}

_MAX_CLAIMS = 8
_SNIPPET_CHARS = 700


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def claim_sentence(answer: str, claim: Claim, max_chars: int = 350) -> str:
    """The sentence ending at the claim's citation marker."""
    boundary = max(
        answer.rfind(". ", 0, claim.span_start),
        answer.rfind("\n", 0, claim.span_start),
        answer.rfind("? ", 0, claim.span_start),
        answer.rfind("! ", 0, claim.span_start),
    )
    sent = answer[boundary + 1 : claim.span_end].strip()
    # strip the trailing `[Page]^0.NN` marker and leading list bullets
    sent = re.sub(r"\[[^\]]+\]\^[\d.]+\s*$", "", sent).strip()
    sent = re.sub(r"^\s*[-*+]\s+", "", sent)
    return sent[-max_chars:]


def _snippet_for(token: str, citations) -> str:
    """Best-matching citation snippet for a `[token]` (same overlap rule as grounding)."""
    t = token.strip().lower()
    for c in citations:
        ct = c.title.strip().lower()
        if t in ct or ct in t:
            return (c.snippet or "")[:_SNIPPET_CHARS]
    return ""


async def verify_claims(client, *, answer: str, claims: list[Claim], citations) -> list[str]:
    """Return one verdict per claim (aligned with `claims`). One gemma call total."""
    verdicts = ["unknown"] * len(claims)
    items: list[tuple[int, str, str]] = []  # (claim_idx, sentence, snippet)
    for i, cl in enumerate(claims[:_MAX_CLAIMS]):
        sent = claim_sentence(answer, cl)
        snip = _snippet_for(cl.citation_token, citations)
        if sent and snip:
            items.append((i, sent, snip))
    if not items:
        return verdicts

    lines = []
    for n, (_, sent, snip) in enumerate(items, start=1):
        lines.append(f"CLAIM {n}: {sent}\nSOURCE {n}: {snip}\n")
    prompt = "\n".join(lines)

    try:
        raw = await client.gemma(prompt, system=_VERIFY_SYSTEM, temperature=0.1)
    except Exception as e:
        log.debug("claim verify call failed", extra={"metadata": {"error": str(e)[:160]}})
        return verdicts
    data = _extract_json(raw) or {}
    for v in data.get("verdicts") or []:
        try:
            n = int(v.get("id", 0))
            verdict = str(v.get("verdict", "")).strip().lower()
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(items) and verdict in VERDICT_WEIGHT:
            claim_idx = items[n - 1][0]
            verdicts[claim_idx] = verdict
    return verdicts


def apply_verdicts(claims: list[Claim], verdicts: list[str]) -> list[dict]:
    """Recalibrate claim confidences in place; return display dicts with verdicts."""
    out: list[dict] = []
    for cl, verdict in zip(claims, verdicts, strict=False):
        weight = VERDICT_WEIGHT.get(verdict, 1.0)
        cl.confidence = max(0.0, min(1.0, cl.confidence * weight))
        out.append({
            "citation": cl.citation_token,
            "confidence": round(cl.confidence, 3),
            "verdict": verdict,
        })
    # claims beyond the zip (verdicts shorter) keep original confidence
    for cl in claims[len(verdicts):]:
        out.append({"citation": cl.citation_token, "confidence": round(cl.confidence, 3),
                    "verdict": "unknown"})
    return out
