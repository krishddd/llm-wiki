"""Multi-provider LLM routing (v4 fleet, v6 open roster).

Lets each text role (summary / reason / fast / solver / vision) and embeddings be
served by a hosted, OpenAI-compatible provider instead of local Ollama. Clone the
repo, set `PROVIDER_<ROLE>` + that provider's key/model in `.env`, run — done.

Roster: Groq (LPU-fast open weights), GitHub Models (gpt-4.1 family), Google
Gemini, OpenAI, Anthropic Claude (OpenAI-compat endpoint), xAI Grok, OpenRouter
(100+ open-source models behind one key), and `custom` — ANY OpenAI-compatible
gateway (vLLM, LM Studio, llama.cpp server, Together, Fireworks, DeepSeek,
Mistral, …) via `CUSTOM_BASE_URL` / `CUSTOM_API_KEY` / `CUSTOM_MODEL`.

Design goals:
- **Opt-in, per role.** `provider_<role>` defaults to "ollama"; nothing changes until
  a role is pointed at a provider AND that provider's key + model are present.
- **Graceful fallback.** Missing key/model → `resolve_*` returns None → caller uses
  Ollama. A provider HTTP failure surfaces as an error the caller already handles.
- **One wire format.** Every provider speaks the OpenAI `/chat/completions` (and
  `/embeddings`) schema with Bearer auth, so a single thin client covers all of
  them. Keys are read from settings (env), never hard-coded.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# Transient HTTP statuses worth retrying: rate-limit + gateway/upstream hiccups
# (hosted free tiers — NVIDIA, OpenRouter — 502/503/504 sporadically).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


async def _post_with_retry(
    http: httpx.AsyncClient,
    url: str,
    *,
    json: dict,
    headers: dict,
    timeout: float,
    retries: int = 2,
    what: str = "request",
) -> httpx.Response:
    """POST that retries transient timeouts / 5xx / 429 with linear backoff.

    Non-transient HTTP errors (4xx auth/validation) surface immediately — retrying a
    401 or 400 is pointless. Raises the last httpx error after exhausting retries."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = await http.post(url, json=json, headers=headers, timeout=timeout)
            if r.status_code in _RETRY_STATUS and attempt < retries:
                log.warning(
                    "provider transient status, retrying",
                    extra={"metadata": {"what": what, "status": r.status_code,
                                        "attempt": attempt + 1, "of": retries + 1}},
                )
                last_err = httpx.HTTPStatusError(
                    f"{r.status_code} transient", request=r.request, response=r)
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            if attempt >= retries:
                break
            log.warning(
                "provider network error, retrying",
                extra={"metadata": {"what": what, "err": type(e).__name__,
                                    "attempt": attempt + 1, "of": retries + 1}},
            )
            await asyncio.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


@dataclass(frozen=True)
class ProviderDef:
    """How to find one provider's config on Settings."""
    base_url_attr: str
    key_attr: str
    chat_model_attr: str
    vision_model_attr: str | None = None
    embed_model_attr: str | None = None
    key_optional: bool = False        # local gateways (vLLM/LM Studio) need no key
    force_max_tokens: bool = False    # Anthropic's compat endpoint requires max_tokens


PROVIDER_DEFS: dict[str, ProviderDef] = {
    "groq": ProviderDef(
        "groq_base_url", "groq_api_key", "groq_model",
        vision_model_attr="groq_vision_model",
    ),
    "github": ProviderDef(
        "github_models_base_url", "github_models_token", "github_models_model",
        vision_model_attr="github_models_vision_model",
    ),
    "gemini": ProviderDef(
        "gemini_base_url", "google_genai_api_key", "gemini_model",
        vision_model_attr="gemini_vision_model",
        embed_model_attr="gemini_embed_model",
    ),
    "openai": ProviderDef(
        "openai_base_url", "openai_api_key", "openai_model",
        vision_model_attr="openai_vision_model",
        embed_model_attr="openai_embed_model",
    ),
    "anthropic": ProviderDef(
        "anthropic_base_url", "anthropic_api_key", "anthropic_model",
        vision_model_attr="anthropic_vision_model",
        force_max_tokens=True,
    ),
    "xai": ProviderDef(
        "xai_base_url", "xai_api_key", "xai_model",
        vision_model_attr="xai_vision_model",
    ),
    "openrouter": ProviderDef(
        "openrouter_base_url", "openrouter_api_key", "openrouter_model",
        vision_model_attr="openrouter_vision_model",
    ),
    "custom": ProviderDef(
        "custom_base_url", "custom_api_key", "custom_model",
        vision_model_attr="custom_vision_model",
        embed_model_attr="custom_embed_model",
        key_optional=True,
    ),
    # NVIDIA build.nvidia.com — OpenAI-compatible NIM gateway. One key covers many
    # open-weight chat models + retrieval embedders. NVIDIA's `nv-embedqa*` embedders
    # require an `input_type` param the plain OpenAI schema omits, so the embed leg
    # attaches `nvidia_embed_input_type` (+ truncate) via `embed_extra` below.
    "nvidia": ProviderDef(
        "nvidia_base_url", "nvidia_api_key", "nvidia_model",
        vision_model_attr="nvidia_vision_model",
        embed_model_attr="nvidia_embed_model",
    ),
}


@dataclass
class ProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str
    force_max_tokens: bool = False
    embed_extra: dict | None = None   # extra /embeddings body params (e.g. NVIDIA input_type)


def _spec(settings, provider: str, model_attr: str | None) -> ProviderSpec | None:
    d = PROVIDER_DEFS[provider]
    if not model_attr:
        return None
    base_url = (getattr(settings, d.base_url_attr, "") or "").rstrip("/")
    api_key = getattr(settings, d.key_attr, "") or ""
    model = getattr(settings, model_attr, "") or ""
    if not base_url or not model or (not api_key and not d.key_optional):
        return None   # incomplete config → caller falls back to Ollama
    return ProviderSpec(
        name=provider, base_url=base_url, api_key=api_key, model=model,
        force_max_tokens=d.force_max_tokens,
    )


def resolve_chat_provider(settings, role: str) -> ProviderSpec | None:
    """Return the provider for a text role, or None to use Ollama.

    `role` ∈ {summary, reason, fast, solver}. Unknown providers / "ollama" → None.
    """
    provider = (getattr(settings, f"provider_{role}", "ollama") or "ollama").lower()
    if provider == "ollama" or provider not in PROVIDER_DEFS:
        return None
    return _spec(settings, provider, PROVIDER_DEFS[provider].chat_model_attr)


def resolve_embed_provider(settings) -> ProviderSpec | None:
    """Return the embeddings provider, or None to use Ollama.

    Supported: gemini, openai, custom (any OpenAI-compatible /embeddings endpoint).
    Groq / GitHub Models / Anthropic / xAI / OpenRouter don't expose embeddings."""
    provider = (getattr(settings, "provider_embed", "ollama") or "ollama").lower()
    if provider == "ollama" or provider not in PROVIDER_DEFS:
        return None
    spec = _spec(settings, provider, PROVIDER_DEFS[provider].embed_model_attr)
    if spec is not None and provider == "nvidia":
        # NVIDIA retrieval embedders (nv-embedqa*) require `input_type`; symmetric ones
        # (bge-m3) accept it harmlessly. `truncate:END` avoids over-length 400s.
        input_type = (getattr(settings, "nvidia_embed_input_type", "") or "").strip()
        extra: dict = {"truncate": "END"}
        if input_type:
            extra["input_type"] = input_type
        spec.embed_extra = extra
    return spec


def resolve_vision_provider(settings) -> ProviderSpec | None:
    """Return the vision provider for the llava (image-caption) role, or None for Ollama."""
    provider = (getattr(settings, "provider_vision", "ollama") or "ollama").lower()
    if provider == "ollama" or provider not in PROVIDER_DEFS:
        return None
    return _spec(settings, provider, PROVIDER_DEFS[provider].vision_model_attr)


def build_chat_payload(
    model: str, prompt: str, system: str | None, temperature: float,
    *, force_max_tokens: bool = False, max_tokens: int = 4096,
) -> dict:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": msgs, "temperature": temperature, "stream": False}
    if force_max_tokens:
        # Anthropic's OpenAI-compat endpoint requires max_tokens explicitly.
        payload["max_tokens"] = max_tokens
    return payload


def _auth_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:  # local OpenAI-compatible gateways (vLLM, LM Studio) need no auth
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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
    payload = build_chat_payload(
        spec.model, prompt, system, temperature, force_max_tokens=spec.force_max_tokens,
    )
    r = await _post_with_retry(
        http, f"{spec.base_url}/chat/completions",
        json=payload, headers=_auth_headers(spec.api_key), timeout=timeout,
        what=f"{spec.name} chat",
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


async def vision_completion(
    http: httpx.AsyncClient,
    spec: ProviderSpec,
    prompt: str,
    image_b64: str,
    *,
    image_mime: str = "image/png",
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """OpenAI-compatible multimodal completion — text prompt + one base64 image.

    Uses the standard `content` array with an `image_url` data URI, which Gemini,
    GitHub Models, and Groq vision models all accept. Raises httpx.HTTPError on failure.
    """
    payload = {
        "model": spec.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            ],
        }],
        "temperature": temperature,
        "stream": False,
    }
    if spec.force_max_tokens:
        payload["max_tokens"] = 4096
    r = await _post_with_retry(
        http, f"{spec.base_url}/chat/completions",
        json=payload, headers=_auth_headers(spec.api_key), timeout=timeout,
        what=f"{spec.name} vision",
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
    body: dict = {"model": spec.model, "input": text}
    if spec.embed_extra:
        body.update(spec.embed_extra)
    r = await _post_with_retry(
        http, f"{spec.base_url}/embeddings",
        json=body, headers=_auth_headers(spec.api_key), timeout=timeout,
        what=f"{spec.name} embed",
    )
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") or []
    if not rows:
        return []
    return list(rows[0].get("embedding") or [])
