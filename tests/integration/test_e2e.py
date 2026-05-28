"""Integration tests — require a live FastAPI + remote Ollama. Marked so normal CI skips them."""
from __future__ import annotations

import os

import httpx
import pytest

API = os.environ.get("API_URL", "http://localhost:8000")
pytestmark = pytest.mark.integration


def test_health_live():
    r = httpx.get(f"{API}/health", timeout=20)
    assert r.status_code == 200
    body = r.json()
    assert body["ollama_reachable"] is True
    assert not body["models_missing"], f"missing models: {body['models_missing']}"


def test_ingest_and_query():
    files = [("files", ("sample.md", b"Docker and FastAPI power the wiki. " * 40, "text/markdown"))]
    r = httpx.post(f"{API}/ingest", files=files, timeout=300)
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 1

    q = httpx.post(f"{API}/query", json={"question": "What powers the wiki?"}, timeout=300)
    assert q.status_code == 200, q.text
    out = q.json()
    assert out["answer"]
    assert "correlation_id" in out
