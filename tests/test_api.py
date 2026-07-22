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

    # `api.py` binds `get_settings` at import (`from .config import get_settings`), so
    # the endpoints resolve the *api-module* name — patching only `cfg.get_settings`
    # (which is `@lru_cache`d) never reaches them, and the startup/endpoints read the
    # real ./wiki. Patch the api-module binding too so the whole app is isolated to the
    # tmp wiki, and clear the cache for any module that reads cfg.get_settings directly.
    cfg.get_settings.cache_clear()
    monkeypatch.setattr(cfg, "get_settings", lambda: tmp_settings)
    monkeypatch.setattr(api_module, "get_settings", lambda: tmp_settings)
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


def _stage_review_page(wiki_dir, name, *, body="Docker is a container runtime.", entity='Docker'):
    """Write a minimal staged review page to the isolated tmp wiki."""
    p = wiki_dir / "review" / f"{name}.md"
    p.write_text(
        "---\n"
        f"title: {name.replace('-', ' ').title()}\n"
        "kind: source\n"
        "confidence: 0.5\n"
        "chunk_count: 1\n"
        f'entity_refs: ["{entity}"]\n'
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return p


def test_review_accept_promotes_and_reindexes(client):
    """POST /review/{id}/accept must move the page to sources/, index it under the
    NEW id (small-to-big chunks, not the old review id), and re-point graph rows."""
    s = api_module.get_settings()
    rev = _stage_review_page(s.wiki_dir, "docker-notes")
    # A graph fact that must follow the page from the review id to the sources id.
    api_module.state.graph._conn.execute(
        "INSERT INTO facts(subject_id,predicate,object_text,source_page,confidence) "
        "VALUES (1,'is','runtime','review/docker-notes.md',0.9)"
    )
    api_module.state.graph._conn.commit()

    r = client.post("/review/docker-notes/accept")
    assert r.status_code == 200, r.text
    assert r.json()["page_path"].endswith("sources/docker-notes.md")

    # File moved.
    assert not rev.exists()
    assert (s.wiki_dir / "sources" / "docker-notes.md").exists()

    # Indexed under the NEW id as sub-chunks; the stale review-id units are purged.
    assert "sources/docker-notes.md#0" in api_module.state.bm25._docs
    assert not any(k.startswith("review/docker-notes.md") for k in api_module.state.bm25._docs)

    # Graph fact re-pointed to the new id (gap #4).
    row = api_module.state.graph._conn.execute("SELECT source_page FROM facts").fetchone()
    assert row[0] == "sources/docker-notes.md"


def test_review_reject_archives_not_deletes(client):
    """POST /review/{id}/reject must archive (reversible), never hard-delete (bug #3)."""
    s = api_module.get_settings()
    rev = _stage_review_page(s.wiki_dir, "junk", body="low quality noise", entity="Nonsense")

    r = client.post("/review/junk/reject")
    assert r.status_code == 200, r.text

    assert not rev.exists()                                   # left review/
    archived = s.wiki_dir / "archive" / "junk.md"
    assert archived.exists()                                 # preserved, not deleted
    assert r.json()["archived_path"].endswith("archive/junk.md")


def test_review_accept_missing_404(client):
    assert client.post("/review/does-not-exist/accept").status_code == 404


def test_review_reject_missing_404(client):
    assert client.post("/review/does-not-exist/reject").status_code == 404


def test_profile_endpoint_returns_contract(client):
    r = client.get("/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["enforcement"] in ("off", "warn", "strict")
    assert "source" in body["profile"]["page_kinds"]
    assert "PERSON" in body["profile"]["entity_types"]


def test_profile_validate_empty_corpus(client):
    r = client.post("/admin/profile/validate")
    assert r.status_code == 200
    assert r.json()["checked"] == 0            # isolated tmp wiki has no source pages


def test_feedback_noise_is_dropped(client):
    r = client.post("/feedback", json={"question": "q", "text": "thanks!"})
    assert r.status_code == 200
    assert r.json()["stored"] is False and r.json()["kind"] == "noise"


def test_feedback_empty_text_400(client):
    assert client.post("/feedback", json={"text": "   "}).status_code == 400


def test_feedback_list_empty(client):
    r = client.get("/feedback")
    assert r.status_code == 200 and r.json()["count"] == 0


def test_feedback_dismiss_missing_404(client):
    assert client.post("/feedback/99999/dismiss").status_code == 404
