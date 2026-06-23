"""Multi-provider LLM routing (v4 fleet).

Lets each text role (summary / reason / fast / solver) and embeddings be served by a
hosted, OpenAI-compatible provider instead of local Ollama — Groq (LPU-fast open
weights), GitHub Models (gpt-4.1 family), or Google Gemini (OpenAI-compat endpoint).

Design goals:
- **Opt-in, per role.** `provider_<role>` defaults to "ollama"; nothing changes until
  a role is pointed at a provider AND that provider's key + model are present.
- **Graceful fallback.** Missing key/model → `resolve_*` returns None → caller uses
  Ollama. A provider HTTP failure surfaces as an error the caller already handles.
- **One wire format.** Groq, GitHub Models, and Gemini all speak the OpenAI
  `/chat/completions` (and `/embeddings`) schema with Bearer auth, so a single thin
  client covers all three. Keys are read from settings (env), never hard-coded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# provider name → (base_url attr, api-key attr, model attr) on Settings.
_CHAT_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "groq":   ("groq_base_url", "groq_api_key", "groq_model"),
    "github": ("github_models_base_url", "github_models_token", "github_models_model"),
    "gemini": ("gemini_base_url", "google_genai_api_key", "gemini_model"),
}


@dataclass
class ProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str


def _spec(settings, provider: str, model_attr: str) -> ProviderSpec | None:
    base_attr, key_attr, default_model_attr = _CHAT_PROVIDERS[provider]
    base_url = (getattr(settings, base_attr, "") or "").rstrip("/")
    api_key = getattr(settings, key_attr, "") or ""
    model = getattr(settings, model_attr, "") or ""
    if not base_url or not api_key or not model:
        return None   # incomplete config → caller falls back to Ollama
    return ProviderSpec(name=provider, base_url=base_url, api_key=api_key, model=model)


def resolve_chat_provider(settings, role: str) -> ProviderSpec | None:
    """Return the provider for a text role, or None to use Ollama.

    `role` ∈ {summary, reason, fast, solver}. Unknown providers / "ollama" → None.
    """
    provider = (getattr(settings, f"provider_{role}", "ollama") or "ollama").lower()
    if provider == "ollama" or provider not in _CHAT_PROVIDERS:
        return None
    # Each provider has one chat model attr; reuse the registry default attr.
    return _spec(settings, provider, _CHAT_PROVIDERS[provider][2])


def resolve_embed_provider(settings) -> ProviderSpec | None:
    """Return the embeddings provider, or None to use Ollama. Only Gemini is supported
    (Groq / GitHub Models do not expose an embeddings endpoint)."""
    provider = (getattr(settings, "provider_embed", "ollama") or "ollama").lower()
    if provider != "gemini":
        return None
    return _spec(settings, "gemini", "gemini_embed_model")


def build_chat_payload(model: str, prompt: str, system: str | None, temperature: float) -> dict:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return {"model": model, "messages": msgs, "temperature": temperature, "stream": False}


def _auth_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def chat_completion(
    http: httpx.AsyncClient,
    spec: ProviderSpec,
    prompt: str,
    system: str | None,
    *,
    temperature: float = 0.3,
    timeout: float = 120.0,
) -> str:
    """OpenAI-compatible chat completion against `spec`. Raises httpx.HTTPError on failure."""
    payload = build_chat_payload(spec.model, prompt, system, temperature)
    r = await http.post(
        f"{spec.base_url}/chat/completions",
        json=payload, headers=_auth_headers(spec.api_key), timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


async def embed_one(
    http: httpx.AsyncClient,
    spec: ProviderSpec,
    text: str,
    *,
    timeout: float = 120.0,
) -> list[float]:
    """OpenAI-compatible embeddings call (Gemini). Raises httpx.HTTPError on failure."""
    r = await http.post(
        f"{spec.base_url}/embeddings",
        json={"model": spec.model, "input": text},
        headers=_auth_headers(spec.api_key), timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") or []
    if not rows:
        return []
    return list(rows[0].get("embedding") or [])
