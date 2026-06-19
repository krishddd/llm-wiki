"""Agentic ingestion — adaptive, content-aware chunking strategy.

The static pipeline chunks every document at a fixed ~6000 chars. That over-chunks
dense technical material (a formula or table gets split from its explanation) and
under-chunks long narrative prose (context fragments unnecessarily). This module
inspects a document's *structure* and *content density* and picks chunk parameters
to fit it:

- **dense** (STEM domain, formulas, or many tables/code blocks) → smaller chunks +
  larger overlap, so notation and its surrounding explanation stay together.
- **narrative** (mostly long-form prose, low structural density) → larger chunks +
  smaller overlap, preserving discourse context.
- **balanced** → the existing defaults.

Heuristic-first (zero LLM cost); optional gemma refinement for ambiguous documents.
The chosen plan is recorded in page frontmatter (`chunk_strategy`) for transparency.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .search.domain import _FORMULA_RE, heuristic_domain

log = logging.getLogger(__name__)

# Clamp bounds — keep the LLM (or a runaway heuristic) from producing absurd sizes.
_MIN_TARGET, _MAX_TARGET = 1500, 9000
_MIN_OVERLAP, _MAX_OVERLAP = 80, 600


@dataclass
class IngestPlan:
    target_chars: int
    overlap_chars: int
    strategy: str          # "dense" | "narrative" | "balanced"
    domain: str
    rationale: str = ""


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _sample_text(elements, *, max_chars: int = 2000) -> str:
    parts: list[str] = []
    for el in elements:
        if getattr(el, "kind", "") in ("text", "heading") and getattr(el, "content", ""):
            parts.append(el.content)
        if sum(len(p) for p in parts) >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def plan_ingest(
    elements,
    *,
    default_target: int = 6000,
    default_overlap: int = 200,
) -> IngestPlan:
    """Heuristic content-aware chunk plan from document structure + a text sample."""
    n_total = len(elements) or 1
    n_table = sum(1 for el in elements if getattr(el, "kind", "") == "table")
    n_code = sum(1 for el in elements if getattr(el, "kind", "") == "code")
    n_text = sum(1 for el in elements if getattr(el, "kind", "") == "text")
    n_heading = sum(1 for el in elements if getattr(el, "kind", "") == "heading")

    structural_density = (n_table + n_code) / n_total
    sample = _sample_text(elements)
    domain = heuristic_domain(sample) or "general"
    has_formula = bool(_FORMULA_RE.search(sample))
    avg_text_len = (
        sum(len(getattr(el, "content", "")) for el in elements if getattr(el, "kind", "") == "text")
        / max(n_text, 1)
    )

    is_dense = (
        domain != "general"
        or has_formula
        or structural_density >= 0.20
        or (n_table + n_code) >= 4
    )
    is_narrative = (
        not is_dense
        and structural_density < 0.05
        and avg_text_len >= 600
        and n_heading <= max(2, n_text // 8)
    )

    if is_dense:
        target = _clamp(3000, _MIN_TARGET, _MAX_TARGET)
        overlap = _clamp(int(target * 0.10), _MIN_OVERLAP, _MAX_OVERLAP)
        strategy = "dense"
        rationale = (
            f"dense content (domain={domain}, formula={has_formula}, "
            f"tables+code={n_table + n_code}, density={structural_density:.2f}) → smaller chunks"
        )
    elif is_narrative:
        target = _clamp(7500, _MIN_TARGET, _MAX_TARGET)
        overlap = _clamp(int(target * 0.02), _MIN_OVERLAP, _MAX_OVERLAP)
        strategy = "narrative"
        rationale = (
            f"narrative prose (avg_text_len={int(avg_text_len)}, density={structural_density:.2f}) "
            f"→ larger chunks"
        )
    else:
        target = _clamp(default_target, _MIN_TARGET, _MAX_TARGET)
        overlap = _clamp(default_overlap, _MIN_OVERLAP, _MAX_OVERLAP)
        strategy = "balanced"
        rationale = "mixed/ambiguous structure → defaults"

    return IngestPlan(
        target_chars=target, overlap_chars=overlap, strategy=strategy, domain=domain, rationale=rationale
    )


_PLAN_SYSTEM = (
    "You are tuning a document chunker. Given a document's structure stats and a text "
    "sample, choose a chunk target size (chars) and overlap (chars). Dense technical or "
    "tabular content wants SMALLER chunks (2000-3500) so formulas/tables stay with their "
    "explanation; long narrative prose wants LARGER chunks (6000-8500). "
    'Reply ONLY JSON: {"target_chars":N,"overlap_chars":N,"strategy":"dense|narrative|balanced","reason":"…"}'
)


async def plan_ingest_llm(client, elements, base_plan: IngestPlan) -> IngestPlan:
    """Optional gemma refinement. Falls back to `base_plan` on any failure."""
    try:
        n_table = sum(1 for el in elements if getattr(el, "kind", "") == "table")
        n_code = sum(1 for el in elements if getattr(el, "kind", "") == "code")
        n_text = sum(1 for el in elements if getattr(el, "kind", "") == "text")
        stats = f"elements: text={n_text} table={n_table} code={n_code}; domain={base_plan.domain}"
        sample = _sample_text(elements, max_chars=1200)
        raw = await client.gemma(f"{stats}\n\nSAMPLE:\n{sample}", system=_PLAN_SYSTEM, temperature=0.1)
        s = re.sub(r"^```(?:json)?\n?", "", (raw or "").strip())
        s = re.sub(r"\n?```$", "", s)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return base_plan
        d = json.loads(m.group(0))
        target = _clamp(int(d.get("target_chars", base_plan.target_chars)), _MIN_TARGET, _MAX_TARGET)
        overlap = _clamp(int(d.get("overlap_chars", base_plan.overlap_chars)), _MIN_OVERLAP, _MAX_OVERLAP)
        strategy = str(d.get("strategy", base_plan.strategy)).strip().lower()
        if strategy not in ("dense", "narrative", "balanced"):
            strategy = base_plan.strategy
        return IngestPlan(
            target_chars=target, overlap_chars=overlap, strategy=strategy,
            domain=base_plan.domain, rationale=f"llm: {str(d.get('reason', ''))[:160]}",
        )
    except Exception as e:
        log.debug("LLM ingest planning failed", extra={"metadata": {"error": str(e)[:120]}})
        return base_plan
