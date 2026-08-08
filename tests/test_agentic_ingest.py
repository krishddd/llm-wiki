"""Agentic ingestion tests — adaptive chunk planning from document structure."""
from __future__ import annotations

from dataclasses import dataclass

from llm_wiki.agentic_ingest import IngestPlan, plan_ingest


@dataclass
class El:
    kind: str
    content: str = ""


def _narrative_doc(paras: int = 12) -> list[El]:
    body = (
        "This is a long passage of ordinary narrative prose that develops an argument "
        "across many sentences without any formulas, tables, or code blocks at all. "
    ) * 6
    return [El("text", body) for _ in range(paras)]


def _dense_doc() -> list[El]:
    els = [El("heading", "Stress analysis")]
    els += [El("text", "The tensile yield strength is computed as $\\sigma = F/A$ for the section.")]
    els += [El("table", "| material | yield |\n|---|---|\n| 6061-T6 | 276 MPa |") for _ in range(3)]
    els += [El("code", "def npv(rate, flows): ...") for _ in range(2)]
    return els


def test_dense_doc_gets_smaller_chunks() -> None:
    plan = plan_ingest(_dense_doc())
    assert plan.strategy == "dense"
    assert plan.target_chars <= 3500
    assert plan.overlap_chars >= 200


def test_narrative_doc_gets_larger_chunks() -> None:
    plan = plan_ingest(_narrative_doc())
    assert plan.strategy == "narrative"
    assert plan.target_chars >= 7000


def test_stem_domain_forces_dense_even_without_tables() -> None:
    els = [El("text", "Solve the integral of x^2 dx and derive the closed form.")]
    plan = plan_ingest(els)
    assert plan.strategy == "dense"
    assert plan.domain == "math"


def test_balanced_fallback_uses_defaults() -> None:
    # A short mixed doc with one table and some prose — neither clearly dense nor narrative.
    els = [El("text", "short note"), El("table", "| a | b |"), El("text", "another short note")]
    plan = plan_ingest(els, default_target=6000, default_overlap=200)
    assert isinstance(plan, IngestPlan)
    # density = 1/3 ≈ 0.33 ≥ 0.20 → classified dense; assert the params stay in bounds.
    assert 1500 <= plan.target_chars <= 9000
    assert 80 <= plan.overlap_chars <= 600


def test_params_always_within_clamp_bounds() -> None:
    for doc in (_dense_doc(), _narrative_doc(), [El("text", "x")]):
        plan = plan_ingest(doc)
        assert 1500 <= plan.target_chars <= 9000
        assert 80 <= plan.overlap_chars <= 600
        assert plan.strategy in ("dense", "narrative", "balanced")
