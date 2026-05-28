"""Generate 2-3 short follow-up questions a user could ask next, given the
current question + answer + retrieved page titles. Cheap qwen call."""
from __future__ import annotations

import logging
import re

from ..llm import OllamaClient

log = logging.getLogger(__name__)


_FOLLOWUP_SYSTEM = (
    "You suggest 2-3 short follow-up questions a curious reader might ask next. "
    "They must be answerable from the provided wiki pages (do NOT invent topics). "
    "Each question is one sentence, ends with '?', and is concrete. "
    "Reply ONLY JSON: {\"follow_ups\":[\"…\",\"…\"]}"
)


def _extract_json_array(s: str) -> list[str]:
    import json
    s = re.sub(r"^```(?:json)?\n?", "", s.strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    arr = data.get("follow_ups") or []
    return [str(x).strip() for x in arr if str(x).strip().endswith("?")][:3]


async def suggest_followups(client: OllamaClient, question: str, answer_summary: str, page_titles: list[str]) -> list[str]:
    if not page_titles or not answer_summary:
        return []
    titles = "\n".join(f"- {t}" for t in page_titles[:8])
    prompt = (
        f"User asked: {question}\n\n"
        f"Brief answer: {answer_summary}\n\n"
        f"Available pages:\n{titles}\n\n"
        "Suggest 2-3 follow-up questions."
    )
    try:
        raw = await client.qwen(prompt, system=_FOLLOWUP_SYSTEM, temperature=0.4)
        return _extract_json_array(raw)
    except Exception as e:
        log.debug("followup suggestion failed", extra={"metadata": {"error": str(e)[:120]}})
        return []
