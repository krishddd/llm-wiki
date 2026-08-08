"""Per-claim confidence parser & calibrator.

We instruct the synth model to embed a confidence at each citation, like:

    Active Inference reduces hallucination [Page Title]^0.92.

This module:
- Parses these `[Page]^0.NN` markers into structured per-claim confidence.
- Strips them out of the user-facing answer if desired.
- Computes a calibrated overall confidence as a weighted aggregate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Match `[anything]^0.92` or `[anything]^.92`. Allow numbers up to 1.0.
_CLAIM_RE = re.compile(r"\[([^\]]+)\]\^(0?\.\d{1,3}|1(?:\.0+)?)")


@dataclass
class Claim:
    citation_token: str   # the page-title or numeric token between [ ]
    confidence: float     # 0.0–1.0
    span_start: int       # char offset of the `[`
    span_end: int         # char offset just past `>` of the `^0.NN`


def parse_claims(answer: str) -> list[Claim]:
    out: list[Claim] = []
    for m in _CLAIM_RE.finditer(answer):
        try:
            conf = float(m.group(2))
        except ValueError:
            continue
        out.append(Claim(citation_token=m.group(1), confidence=max(0.0, min(1.0, conf)),
                         span_start=m.start(), span_end=m.end()))
    return out


def strip_confidence_markers(answer: str) -> str:
    """Remove `^0.NN` so the user-facing answer is clean. Citations stay intact."""
    return _CLAIM_RE.sub(lambda m: f"[{m.group(1)}]", answer)


def aggregate_confidence(claims: list[Claim], floor: float = 0.0, ceiling: float = 1.0) -> float:
    """Aggregate per-claim confidences into one number.

    Strategy: weighted toward the lowest claim (a single shaky claim drags the
    overall down). Specifically: 0.6 × min + 0.4 × mean. Clamped to [floor, ceiling].
    """
    if not claims:
        return ceiling
    confs = [c.confidence for c in claims]
    aggregate = 0.6 * min(confs) + 0.4 * (sum(confs) / len(confs))
    return max(floor, min(ceiling, aggregate))
