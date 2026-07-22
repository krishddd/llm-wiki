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
    resolve_vision_provider,
    vision_completion,
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
        provider_vision="ollama", groq_vision_model="",
        github_models_vision_model="openai/gpt-4.1-mini", gemini_vision_model="gemini-2.5-flash-lite",
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


def test_three_provider_fleet_resolves_together() -> None:
    # Groq (fast) + GitHub Models (reason) + Gemini (summary + embed + vision), all at once.
    s = _settings(
        provider_fast="groq", groq_api_key="gsk_x",
        provider_reason="github", github_models_token="ghp_x",
        provider_summary="gemini", google_genai_api_key="AIz_x",
        provider_embed="gemini", provider_vision="gemini",
    )
    assert resolve_chat_provider(s, "fast").name == "groq"
    assert resolve_chat_provider(s, "reason").name == "github"
    assert resolve_chat_provider(s, "summary").name == "gemini"
    assert resolve_embed_provider(s).name == "gemini"          # embeddings → Gemini
    assert resolve_vision_provider(s).name == "gemini"
    # A role left on ollama stays on ollama.
    assert resolve_chat_provider(s, "solver") is None


def test_embeddings_reject_non_gemini_providers() -> None:
    # Groq / GitHub have no embeddings endpoint → must fall back to Ollama (None).
    assert resolve_embed_provider(_settings(provider_embed="groq", groq_api_key="gsk_x")) is None
    assert resolve_embed_provider(_settings(provider_embed="github", github_models_token="ghp_x")) is None


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


def test_vision_provider_resolution() -> None:
    assert resolve_vision_provider(_settings(provider_vision="ollama")) is None
    # Groq vision model unset → falls back to Ollama.
    assert resolve_vision_provider(_settings(provider_vision="groq", groq_api_key="gsk_x")) is None
    s = _settings(provider_vision="gemini", google_genai_api_key="AIz_x")
    spec = resolve_vision_provider(s)
    assert spec is not None and spec.name == "gemini" and spec.model == "gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_vision_completion_sends_image_url() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a red square"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        spec = ProviderSpec("gemini", "https://x/openai", "AIz", "gemini-2.5-flash-lite")
        out = await vision_completion(http, spec, "describe", "BASE64DATA", image_mime="image/png")

    assert out == "a red square"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,BASE64DATA"


@pytest.mark.asyncio
async def test_embed_one_offline() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
    ) as http:
        spec = ProviderSpec("gemini", "https://x/openai", "AIz", "text-embedding-004")
        vec = await embed_one(http, spec, "text")
    assert vec == [0.1, 0.2, 0.3]


# ───── v6 open roster: openai / anthropic / xai / openrouter / custom ─────


def _settings6(**kw):
    base = dict(
        openai_base_url="https://api.openai.com/v1", openai_api_key="",
        openai_model="gpt-4.1-mini", openai_vision_model="gpt-4.1-mini",
        openai_embed_model="text-embedding-3-small",
        anthropic_base_url="https://api.anthropic.com/v1", anthropic_api_key="",
        anthropic_model="claude-sonnet-5", anthropic_vision_model="claude-sonnet-5",
        xai_base_url="https://api.x.ai/v1", xai_api_key="", xai_model="grok-4",
        xai_vision_model="",
        openrouter_base_url="https://openrouter.ai/api/v1", openrouter_api_key="",
        openrouter_model="meta-llama/llama-3.3-70b-instruct", openrouter_vision_model="",
        custom_base_url="", custom_api_key="", custom_model="",
        custom_vision_model="", custom_embed_model="",
    )
    base.update(kw)
    return _settings(**base)


def test_openai_and_anthropic_routes() -> None:
    s = _settings6(provider_reason="anthropic", anthropic_api_key="sk-ant-x",
                   provider_summary="openai", openai_api_key="sk-x")
    reason = resolve_chat_provider(s, "reason")
    assert reason is not None and reason.name == "anthropic"
    assert reason.model == "claude-sonnet-5"
    assert reason.force_max_tokens is True   # Anthropic compat needs max_tokens
    summary = resolve_chat_provider(s, "summary")
    assert summary is not None and summary.name == "openai"
    assert summary.force_max_tokens is False


def test_xai_and_openrouter_routes() -> None:
    s = _settings6(provider_fast="xai", xai_api_key="xai-x",
                   provider_solver="openrouter", openrouter_api_key="sk-or-x")
    assert resolve_chat_provider(s, "fast").model == "grok-4"
    assert resolve_chat_provider(s, "solver").name == "openrouter"


def test_custom_gateway_without_key() -> None:
    # Local vLLM / LM Studio gateways need no API key — base_url + model suffice.
    s = _settings6(provider_reason="custom",
                   custom_base_url="http://localhost:8001/v1", custom_model="qwen2.5-32b")
    spec = resolve_chat_provider(s, "reason")
    assert spec is not None and spec.name == "custom" and spec.api_key == ""
    # But a missing base_url still falls back to Ollama.
    assert resolve_chat_provider(_settings6(provider_reason="custom"), "reason") is None


def test_embeddings_openai_and_custom() -> None:
    s = _settings6(provider_embed="openai", openai_api_key="sk-x")
    assert resolve_embed_provider(s).model == "text-embedding-3-small"
    s2 = _settings6(provider_embed="custom", custom_base_url="http://localhost:8001/v1",
                    custom_embed_model="bge-m3")
    assert resolve_embed_provider(s2).model == "bge-m3"
    # Anthropic / xAI / OpenRouter have no embeddings endpoint.
    assert resolve_embed_provider(_settings6(provider_embed="anthropic", anthropic_api_key="k")) is None
    assert resolve_embed_provider(_settings6(provider_embed="xai", xai_api_key="k")) is None


def test_anthropic_payload_includes_max_tokens() -> None:
    p = build_chat_payload("claude-sonnet-5", "hi", "sys", 0.3, force_max_tokens=True)
    assert p["max_tokens"] == 4096
    p2 = build_chat_payload("gpt-4.1-mini", "hi", None, 0.3)
    assert "max_tokens" not in p2


@pytest.mark.asyncio
async def test_chat_completion_no_auth_header_when_keyless() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        spec = ProviderSpec("custom", "http://localhost:8001/v1", "", "local-model")
        out = await chat_completion(http, spec, "hi", None)
    assert out == "ok"
    assert captured["auth"] is None


# ── NVIDIA build.nvidia.com provider ─────────────────────────────────────────

def _nv_settings(**kw):
    base = dict(
        provider_summary="nvidia", provider_reason="nvidia", provider_fast="nvidia",
        provider_solver="nvidia", provider_embed="nvidia", provider_vision="nvidia",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key="nvapi-xxx",
        nvidia_model="meta/llama-3.3-70b-instruct",
        nvidia_embed_model="nvidia/nv-embedqa-e5-v5",
        nvidia_vision_model="meta/llama-3.2-11b-vision-instruct",
        nvidia_embed_input_type="query",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_nvidia_resolves_all_text_roles_to_one_model() -> None:
    s = _nv_settings()
    for role in ("summary", "reason", "fast", "solver"):
        spec = resolve_chat_provider(s, role)
        assert spec is not None and spec.name == "nvidia"
        assert spec.model == "meta/llama-3.3-70b-instruct"
        assert spec.base_url == "https://integrate.api.nvidia.com/v1"


def test_nvidia_embed_spec_carries_input_type() -> None:
    spec = resolve_embed_provider(_nv_settings())
    assert spec is not None and spec.model == "nvidia/nv-embedqa-e5-v5"
    assert spec.embed_extra == {"truncate": "END", "input_type": "query"}


def test_nvidia_embed_input_type_omitted_when_blank() -> None:
    spec = resolve_embed_provider(_nv_settings(nvidia_embed_input_type=""))
    assert spec.embed_extra == {"truncate": "END"}


def test_nvidia_missing_key_falls_back_to_ollama() -> None:
    assert resolve_chat_provider(_nv_settings(nvidia_api_key=""), "reason") is None


@pytest.mark.asyncio
async def test_nvidia_embed_one_sends_input_type_in_body() -> None:
    spec = resolve_embed_provider(_nv_settings())
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        vec = await embed_one(http, spec, "docker notes")
    assert vec == [0.1, 0.2]
    assert captured["model"] == "nvidia/nv-embedqa-e5-v5"
    assert captured["input"] == "docker notes"
    assert captured["input_type"] == "query"
    assert captured["truncate"] == "END"
