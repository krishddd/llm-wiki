"""Lightweight Qwen-backed intent router. Not used for the primary REST endpoints
(which are explicit), but exposed as /route for free-form CLI agents."""
from __future__ import annotations

import json
import re

from .llm import OllamaClient, get_client

ROUTE_SYSTEM = (
    "You are a routing agent for an LLM-Wiki. Classify the user input as one of: INGEST, QUERY, LINT, SCHEMA_UPDATE. "
    'Reply ONLY JSON: {"action":"INGEST|QUERY|LINT|SCHEMA_UPDATE","args":{...}}'
)


async def route(user_input: str, client: OllamaClient | None = None) -> dict:
    c = client or get_client()
    raw = await c.qwen(user_input, system=ROUTE_SYSTEM, temperature=0.1)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"action": "QUERY", "args": {"question": user_input}}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "QUERY", "args": {"question": user_input}}
