"""FastAPI smoke tests — correlation-id middleware + health + review stub.

We patch the module-level `get_client` to return the FakeOllama fixture, and
we call the lifespan startup manually since TestClient uses sync lifecycle.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src import api as api_module


@pytest.fixture
def client(monkeypatch, fake_ollama, tmp_settings):
    from src import config as cfg
    monkeypatch.setattr(cfg, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(api_module, "get_client", lambda: fake_ollama)
    with TestClient(api_module.app) as c:
        yield c


def test_health_returns_correlation_id(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Correlation-ID" in r.headers
    assert r.headers["X-Correlation-ID"].startswith("COR-")
    body = r.json()
    assert body["ollama_reachable"] is True
    assert body["models_missing"] == []


def test_ingest_roundtrip(client):
    f = ("sample.md", io.BytesIO(b"Docker and FastAPI integration notes. " * 30), "text/markdown")
    r = client.post("/ingest", files=[("files", f)])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] == 1
    assert data["results"][0]["is_live"] is True


def test_review_list_empty(client):
    r = client.get("/review")
    assert r.status_code == 200
    assert r.json()["count"] == 0
