"""Multi-provider LLM routing tests — resolution + OpenAI-compat wire format (offline)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src.providers import (
    ProviderSpec,
    build_chat_payload,
    chat_completion,
    embed_one,
    resolve_chat_provider,
    resolve_embed_provider,
)


def _settings(**kw):
    base = dict(
        provider_summary="ollama", provider_reason="ollama", provider_fast="ollama",
        provider_solver="ollama", provider_embed="ollama",
        groq_base_url="https://api.groq.com/openai/v1", groq_api_key="", groq_model="llama-3.3-70b-versatile",
        github_models_base_url="https://models.github.ai/inference", github_models_token="",
        github_models_model="openai/gpt-4.1-mini",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai", google_genai_api_key="",
        gemini_model="gemini-2.5-flash-lite", gemini_embed_model="text-embedding-004",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_default_all_roles_ollama() -> None:
    s = _settings()
    for role in ("summary", "reason", "fast", "solver"):
        assert resolve_chat_provider(s, role) is None


def test_groq_fast_route() -> None:
    s = _settings(provider_fast="groq", groq_api_key="gsk_x")
    spec = resolve_chat_provider(s, "fast")
    assert spec is not None
    assert spec.name == "groq"
    assert spec.model == "llama-3.3-70b-versatile"
    assert spec.base_url == "https://api.groq.com/openai/v1"


def test_github_reason_route() -> None:
    s = _settings(provider_reason="github", github_models_token="ghp_x")
    spec = resolve_chat_provider(s, "reason")
    assert spec is not None and spec.name == "github" and "gpt-4.1" in spec.model


def test_missing_key_falls_back_to_ollama() -> None:
    s = _settings(provider_reason="github")   # token unset
    assert resolve_chat_provider(s, "reason") is None


def test_unknown_provider_is_ollama() -> None:
    assert resolve_chat_provider(_settings(provider_fast="bogus"), "fast") is None


def test_embed_provider_only_gemini() -> None:
    assert resolve_embed_provider(_settings(provider_embed="ollama")) is None
    assert resolve_embed_provider(_settings(provider_embed="groq")) is None       # no Groq embeddings
    s = _settings(provider_embed="gemini", google_genai_api_key="AIz_x")
    spec = resolve_embed_provider(s)
    assert spec is not None and spec.model == "text-embedding-004"


def test_build_chat_payload_shape() -> None:
    p = build_chat_payload("m", "hi", "sys", 0.5)
    assert p["model"] == "m" and p["temperature"] == 0.5 and p["stream"] is False
    assert p["messages"][0] == {"role": "system", "content": "sys"}
    assert p["messages"][-1] == {"role": "user", "content": "hi"}


def test_build_chat_payload_no_system() -> None:
    p = build_chat_payload("m", "hi", None, 0.3)
    assert len(p["messages"]) == 1 and p["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_completion_offline() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello world"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        spec = ProviderSpec("groq", "https://api.groq.com/openai/v1", "gsk_test", "llama-3.3-70b-versatile")
        out = await chat_completion(http, spec, "hi", "sys", temperature=0.2)

    assert out == "hello world"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer gsk_test"
    assert captured["body"]["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_chat_completion_empty_choices() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))) as http:
        spec = ProviderSpec("groq", "https://x/v1", "k", "m")
        assert await chat_completion(http, spec, "hi", None) == ""


@pytest.mark.asyncio
async def test_embed_one_offline() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
    ) as http:
        spec = ProviderSpec("gemini", "https://x/openai", "AIz", "text-embedding-004")
        vec = await embed_one(http, spec, "text")
    assert vec == [0.1, 0.2, 0.3]
