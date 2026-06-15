"""Feedback-driven query rewriter.

Takes the SCA's `missing_aspects` + `suggested_queries` and produces concrete
search queries aimed at filling the gap. Different from
`src.search.multi_query.paraphrase` (which rephrases the SAME question);
this one targets DIFFERENT information.
"""
from __future__ import annotations

import json
import logging
import re

from ..llm import OllamaClient

log = logging.getLogger(__name__)


REWRITE_SYSTEM = (
    "You are a query rewriter for a wiki search system. The previous retrieval "
    "did not cover certain aspects of the user's question. Your job: emit 2–4 "
    "NEW search queries that target the MISSING aspects.\n\n"
    "Rules:\n"
    "- Each query must focus on what is MISSING, not the whole original question.\n"
    "- Use concrete nouns and identifiers when present (entity names, IDs, codes).\n"
    "- Do NOT repeat queries already tried (you will be given the previous list).\n"
    "- Prefer short keyword-style queries (4-10 words) over full sentences.\n"
    "- If the SCA already suggested good queries, you may keep/refine them but "
    "deduplicate against the previous list.\n\n"
    'Reply ONLY JSON: {"queries":["...","..."]}'
)


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


def _dedupe_against(candidates: list[str], seen: list[str]) -> list[str]:
    seen_norm = {q.strip().lower() for q in seen}
    out: list[str] = []
    out_norm: set[str] = set()
    for c in candidates:
        n = c.strip().lower()
        if not n or n in seen_norm or n in out_norm:
            continue
        out.append(c.strip())
        out_norm.add(n)
    return out


async def rewrite_for_gaps(
    client: OllamaClient,
    *,
    original_question: str,
    missing_aspects: list[str],
    previous_queries: list[str],
    suggested_queries: list[str] | None = None,
    max_queries: int = 4,
) -> list[str]:
    """Return up to `max_queries` gap-targeted search queries.

    Falls back to the SCA's `suggested_queries` (deduped) if the LLM call fails.
    """
    suggested_queries = suggested_queries or []

    if not missing_aspects:
        # Nothing missing → just return SCA suggestions deduped.
        return _dedupe_against(suggested_queries, previous_queries)[:max_queries]

    missing_block = "\n".join(f"- {m}" for m in missing_aspects[:6])
    prev_block = "\n".join(f"- {q}" for q in previous_queries[:10]) or "(none)"
    sugg_block = "\n".join(f"- {q}" for q in suggested_queries[:6]) or "(none)"

    prompt = (
        f"ORIGINAL QUESTION:\n{original_question}\n\n"
        f"MISSING ASPECTS:\n{missing_block}\n\n"
        f"PREVIOUSLY TRIED QUERIES (DO NOT REPEAT):\n{prev_block}\n\n"
        f"SCA-SUGGESTED QUERIES (may refine):\n{sugg_block}\n\n"
        "Emit the JSON now."
    )

    try:
        raw = await client.qwen(prompt, system=REWRITE_SYSTEM, temperature=0.2)
        data = _extract_json(raw) or {}
        queries = data.get("queries") or []
        cleaned = [str(q).strip() for q in queries if str(q).strip()]
    except Exception as e:
        log.debug("rewrite_for_gaps LLM failed; using SCA suggestions",
                  extra={"metadata": {"error": str(e)[:120]}})
        cleaned = list(suggested_queries)

    # Always also consider SCA suggestions as a safety net.
    merged = _dedupe_against(cleaned + list(suggested_queries), previous_queries)
    return merged[:max_queries]
