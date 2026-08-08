"""Adaptive model-routing unit tests — domain detection + solver routing + think-strip.

Covers the VibeThinker integration seam: quantitative questions (maths, economics,
science, engineering / industrial materials) must route to the reasoning specialist,
while plain-English recall stays on the general synthesizer.
"""
from __future__ import annotations

import pytest

from llm_wiki.llm import strip_think
from llm_wiki.search.domain import heuristic_domain, needs_solver


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the Free Energy Principle?", "general"),
        ("Who founded Anthropic?", "general"),
        ("Solve the integral of x^2 dx", "math"),
        ("Prove that the sum of the first n integers is n(n+1)/2", "math"),
        ("Compute the NPV at a discount rate of 8%", "economics"),
        ("What is the price elasticity of demand here?", "economics"),
        ("Calculate the enthalpy change for this reaction", "science"),
        ("What is the tensile yield strength of 6061-T6 aluminium?", "engineering"),
    ],
)
def test_heuristic_domain(question: str, expected: str) -> None:
    assert (heuristic_domain(question) or "general") == expected


def test_needs_solver_routes_quantitative_questions() -> None:
    assert needs_solver("Compute the NPV at a discount rate of 8%") is True
    assert needs_solver("Solve 3x + 2 = 11 for x") is True


def test_needs_solver_skips_plain_english() -> None:
    assert needs_solver("Who founded Anthropic?") is False
    assert needs_solver("Summarize the history of the wiki project") is False


def test_needs_solver_triggers_on_formula_in_context() -> None:
    # Plain-English question, but the retrieved page is formula-heavy → still route.
    assert needs_solver("Explain this page", "the model gives $E=mc^2$ where ...") is True


def test_strip_think_removes_reasoning_trace() -> None:
    assert strip_think("<think>long chain of thought</think>Final: 42") == "Final: 42"


def test_strip_think_handles_unclosed_tag() -> None:
    # Truncated generation: opening tag, no close. Contract is "never leak a raw
    # <think> tag" — best-effort trailing text is returned for the formatter to salvage.
    out = strip_think("<think>dangling reasoning with no close")
    assert "<think>" not in out
    assert out == "dangling reasoning with no close"


def test_strip_think_passthrough_when_no_trace() -> None:
    assert strip_think("Just a plain answer.") == "Just a plain answer."
