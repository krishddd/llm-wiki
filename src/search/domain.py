"""Domain / cognition detection — orthogonal to retrieval *intent*.

`intent.py` decides the retrieval SHAPE (factual / multi_hop / synthesis /
exhaustive). This module decides the COGNITION required: is the task plain-English
recall, or verifiable quantitative reasoning over maths / economics / science /
engineering content?

That signal drives **adaptive model routing**: reasoning-heavy questions are sent
to a specialist reasoner (VibeThinker — strong on AIME-class maths & STEM, weak on
broad knowledge), while everything else stays on the general synthesizer (qwen).

Heuristic-first (zero LLM cost on obvious cases), gemma fallback only when the
regex signal is ambiguous AND the caller asks for it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ..llm import OllamaClient

log = logging.getLogger(__name__)

Domain = Literal["general", "math", "science", "economics", "engineering"]


@dataclass
class DomainProfile:
    domain: Domain
    needs_solver: bool   # route to the reasoning specialist
    rationale: str = ""


# ── Heuristic signals ───────────────────────────────────────────────────────
# LaTeX / formula / equation markers — the strongest "this is quantitative" tell.
_FORMULA_RE = re.compile(
    r"(\$[^$]+\$|\\frac|\\sum|\\int|\\sqrt|\\partial|\\nabla|\\alpha|\\beta|"
    r"\\theta|\\sigma|=\s*-?\d|[<>]=?\s*-?\d|\b\d+\s*[+\-*/^]\s*\d)",
)
_MATH_RE = re.compile(
    r"\b(solve|derive|prove|theorem|lemma|integral|derivative|differential equation|"
    r"matrix|eigen(value|vector)|probability|combinatoric|factorial|polynomial|"
    r"optimi[sz]e|minimi[sz]e|maximi[sz]e|gradient|calculus|algebra|geometry)\b",
    re.IGNORECASE,
)
_ECON_RE = re.compile(
    r"\b(NPV|IRR|ROI|elasticity|equilibrium|supply and demand|marginal (cost|utility|revenue)|"
    r"GDP|inflation|discount rate|present value|opportunity cost|cash ?flow|"
    r"amorti[sz]ation|compound interest|profit margin|break[- ]even)\b",
    re.IGNORECASE,
)
_SCIENCE_RE = re.compile(
    r"\b(stoichiometr|mole?s?\b|molar|reaction|enthalpy|entropy|velocity|acceleration|"
    r"momentum|wavelength|frequency|voltage|current|resistance|equilibrium constant|"
    r"half[- ]life|concentration|pH\b|oxidation|thermodynamic|quantum)\b",
    re.IGNORECASE,
)
_ENGINEERING_RE = re.compile(
    r"\b(tensile|yield strength|stress|strain|shear|torque|load[- ]bearing|"
    r"young'?s modulus|fatigue|elastic modulus|cross[- ]section|moment of inertia|"
    r"thermal conductivity|tolerance|material propert(y|ies)|alloy|composite|"
    r"flow rate|pressure drop|heat transfer)\b",
    re.IGNORECASE,
)

# Order matters: more specific domains win over bare "math".
_DOMAIN_RES: list[tuple[Domain, re.Pattern[str]]] = [
    ("economics", _ECON_RE),
    ("engineering", _ENGINEERING_RE),
    ("science", _SCIENCE_RE),
    ("math", _MATH_RE),
]


def heuristic_domain(text: str) -> Domain | None:
    """Cheap rule-based domain detection. Returns None when no signal is present."""
    t = text or ""
    has_formula = bool(_FORMULA_RE.search(t))
    for dom, rx in _DOMAIN_RES:
        if rx.search(t):
            return dom
    # Formula markers with no keyword match still imply a quantitative ("math") task.
    if has_formula:
        return "math"
    return None


def needs_solver(question: str, context: str = "") -> bool:
    """True when the task is verifiable reasoning (route to the specialist).

    We look at the question first (cheap) and the head of the retrieved context
    second — a plain-English question over a maths-heavy page still benefits from
    the reasoner.
    """
    if heuristic_domain(question) is not None:
        return True
    return bool(context and _FORMULA_RE.search(context[:2000]))


_DOMAIN_SYSTEM = (
    "Classify the dominant subject of the text into ONE class:\n"
    "- general: plain-language knowledge, no quantitative reasoning needed\n"
    "- math: maths / proofs / calculation\n"
    "- science: physics / chemistry / biology with quantitative content\n"
    "- economics: finance / economic modelling / quantitative business\n"
    "- engineering: materials / mechanical / industrial calculations\n"
    'Reply ONLY JSON: {"domain":"general|math|science|economics|engineering","reason":"…"}'
)


async def classify_domain_llm(client: OllamaClient, text: str) -> Domain:
    try:
        raw = await client.summarize(text[:2000], system=_DOMAIN_SYSTEM, temperature=0.1)
        s = re.sub(r"^```(?:json)?\n?", "", (raw or "").strip())
        s = re.sub(r"\n?```$", "", s)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            v = str(d.get("domain", "general")).strip().lower()
            if v in ("general", "math", "science", "economics", "engineering"):
                return v  # type: ignore[return-value]
    except Exception as e:
        log.debug("LLM domain classification failed", extra={"metadata": {"error": str(e)[:120]}})
    return "general"


async def domain_for(
    client: OllamaClient,
    text: str,
    *,
    use_llm_fallback: bool = False,
) -> DomainProfile:
    """Detect the domain of `text`. Heuristic first; gemma fallback only if asked."""
    dom = heuristic_domain(text)
    rationale = "heuristic"
    if dom is None:
        if use_llm_fallback:
            dom = await classify_domain_llm(client, text)
            rationale = "llm"
        else:
            dom = "general"
            rationale = "default"
    return DomainProfile(domain=dom, needs_solver=dom != "general", rationale=rationale)
