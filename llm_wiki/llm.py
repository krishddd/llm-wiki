"""Async Ollama client. Per-role helpers + timeout-triggered fallback to model_fast.

All calls hit the remote `OLLAMA_HOST` via `/api/chat` and `/api/embeddings`.
If `model_fast == model_reason`, the fallback is skipped and the original error propagates.
"""
from __future__ import annotations

import base64
import contextlib
import logging
import math
import re
from pathlib import Path

import httpx

from .config import Settings, get_settings

log = logging.getLogger(__name__)


def truncate_mrl(vec: list[float], dims: int) -> list[float]:
    """Matryoshka truncation: keep the leading `dims` dimensions and L2-renormalize.

    MRL-trained embedders concentrate semantics in the leading dimensions, so a
    truncated + renormalized prefix stays a valid unit vector in a lower-dimensional
    space with near-identical relative distances. No-op when `dims<=0`, the vector is
    empty, or it is already at/under `dims` (never pads)."""
    if dims <= 0 or not vec or len(vec) <= dims:
        return vec
    head = vec[:dims]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]

# Reasoning models (VibeThinker, qwen-thinking, DeepSeek-R1, …) emit an explicit
# chain-of-thought before the final answer. Strip it so downstream JSON parsing /
# formatting sees only the conclusion.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>…</think> reasoning traces; return the trailing answer."""
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    # Some models leave a dangling/unclosed <think> with no closing tag — keep only
    # whatever follows the LAST opening tag in that case.
    if "<think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        if idx != -1:
            cleaned = cleaned[idx + len("</think>"):]
        else:
            parts = re.split(r"<think>", cleaned, flags=re.IGNORECASE)
            cleaned = parts[-1]
    return cleaned.strip()


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    _EMBED_CACHE_MAX = 512

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        self._client = client or httpx.AsyncClient(timeout=self.settings.llm_timeout)
        # Simple FIFO-evicted cache. Insertion-ordered dict gives us O(1) eviction.
        self._embed_cache: dict[str, list[float]] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self.settings.ollama_host}/api/tags")
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    async def _chat(self, model: str, prompt: str, system: str | None, *, temperature: float = 0.3, images: list[str] | None = None, timeout: float | None = None, retries: int = 2) -> str:
        """Single chat call. Retries `retries` times on ReadTimeout / transient errors.

        Ollama silently reloads a model when another model is requested — the reload
        can take 30-60s and will trip one timeout. A retry after the model is warm
        almost always succeeds, so we absorb it here rather than exploding the whole
        ingest pipeline.
        """
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        user_msg: dict = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        msgs.append(user_msg)
        payload = {"model": model, "messages": msgs, "stream": False, "options": {"temperature": temperature}}
        effective_timeout = timeout or self.settings.llm_timeout

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = await self._client.post(
                    f"{self.settings.ollama_host}/api/chat", json=payload, timeout=effective_timeout
                )
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "")
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as e:
                last_err = e
                # informative message even though str(ReadTimeout) is empty
                log.warning(
                    "ollama timeout, retrying",
                    extra={"metadata": {
                        "model": model, "attempt": attempt + 1, "of": retries + 1,
                        "err_type": type(e).__name__, "timeout_s": effective_timeout,
                    }},
                )
                continue
            except httpx.HTTPError as e:
                # Non-timeout HTTP errors: surface immediately, no retry.
                raise OllamaError(f"chat failed for {model}: {type(e).__name__}: {e!s}") from e

        # Exhausted retries on timeouts.
        assert last_err is not None
        raise OllamaError(
            f"chat failed for {model}: {type(last_err).__name__} after {retries + 1} attempts "
            f"(timeout={effective_timeout}s). Likely Ollama model-swap — set OLLAMA_KEEP_ALIVE=60m "
            f"and OLLAMA_MAX_LOADED_MODELS=1."
        ) from last_err

    async def _role_chat(
        self, role: str, ollama_model: str, prompt: str, system: str | None,
        *, temperature: float = 0.3, timeout: float | None = None,
    ) -> str:
        """Dispatch a text role to its configured provider, or Ollama by default.

        Hosted providers (Groq / GitHub / Gemini / OpenAI / Anthropic / xAI /
        OpenRouter / custom) are OpenAI-compatible; a provider HTTP error is
        re-raised as OllamaError so existing role fallbacks (e.g. reason→fast)
        still apply. Missing key/model → transparent Ollama path.
        """
        from .providers import chat_completion, resolve_chat_provider
        spec = resolve_chat_provider(self.settings, role)
        if spec is None:
            return await self._chat(ollama_model, prompt, system, temperature=temperature, timeout=timeout)
        try:
            return await chat_completion(
                self._client, spec, prompt, system,
                temperature=temperature, timeout=timeout or self.settings.llm_timeout,
            )
        except httpx.HTTPError as e:
            raise OllamaError(f"{spec.name} chat failed for {spec.model}: {type(e).__name__}: {e!s}") from e

    # ── Role methods ─────────────────────────────────────────────────────────
    # Each method is a ROLE, not a model. The concrete model is chosen at call
    # time from settings (`model_<role>` on Ollama, or the `provider_<role>`
    # hosted model). Swapping models is pure configuration — no code changes.
    #   summarize → summary role   reason → deep-reasoning/synthesis role
    #   fast      → fast-agent role  vision → image-caption role
    #   solver    → quantitative specialist   embed → embeddings

    async def summarize(self, prompt: str, system: str | None = None, *, temperature: float = 0.4) -> str:
        return await self._role_chat("summary", self.settings.model_summary, prompt, system, temperature=temperature)

    async def reason(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        try:
            return await self._role_chat("reason", self.settings.model_reason, prompt, system, temperature=temperature)
        except OllamaError as e:
            # Only fall back if model_fast is a genuinely different model; otherwise re-raise
            # so the caller's own fallback (e.g. ingest's merge-concat) kicks in instead of
            # retrying the same overloaded model.
            if self.settings.model_fast and self.settings.model_fast != self.settings.model_reason:
                log.warning(
                    f"reason role failed, falling back to {self.settings.model_fast}",
                    extra={"metadata": {"error": str(e)}},
                )
                return await self.fast(prompt, system, temperature=temperature)
            raise

    async def fast(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        # Fast-agent role: model_fast on Ollama, or the provider_fast hosted model.
        return await self._role_chat(
            "fast", self.settings.model_fast, prompt, system,
            temperature=temperature, timeout=self.settings.llm_fast_timeout,
        )

    async def solver(self, prompt: str, system: str | None = None, *, temperature: float | None = None) -> str:
        """Reasoning specialist (VibeThinker). Returns the answer with any
        <think> trace stripped. Caller is responsible for routing only quantitative
        tasks here — VibeThinker is weak on broad-knowledge / recall.

        Raises OllamaError if `model_solver` is unset or the model isn't served;
        callers should catch and fall back to reason-role synthesis.
        """
        spec_provider = self.settings.provider_solver and self.settings.provider_solver.lower() != "ollama"
        if not self.settings.model_solver and not spec_provider:
            raise OllamaError("model_solver is not configured")
        temp = self.settings.solver_temperature if temperature is None else temperature
        raw = await self._role_chat("solver", self.settings.model_solver, prompt, system, temperature=temp)
        return strip_think(raw)

    async def vision(self, prompt: str, image_path: str | Path) -> str:
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        from .providers import resolve_vision_provider, vision_completion
        spec = resolve_vision_provider(self.settings)
        if spec is not None:
            mime = "image/jpeg" if Path(image_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
            try:
                return await vision_completion(
                    self._client, spec, prompt, img_b64,
                    image_mime=mime, timeout=self.settings.llm_timeout,
                )
            except httpx.HTTPError as e:
                raise OllamaError(f"{spec.name} vision failed for {spec.model}: {type(e).__name__}: {e!s}") from e
        return await self._chat(self.settings.model_vision, prompt, None, images=[img_b64])

    async def embed(self, text: str, *, model: str | None = None) -> list[float]:
        # A `model` override (used by the STEM-routed dense index) always means a
        # specific Ollama embedder; only the default path may route to a provider.
        from .providers import embed_one, resolve_embed_provider
        embed_spec = resolve_embed_provider(self.settings) if model is None else None
        embed_model = model or (embed_spec.model if embed_spec else self.settings.model_embed)
        # Matryoshka truncation applies only to the DEFAULT embedder (model is None):
        # a `model` override means a specific non-MRL embedder (e.g. bge-m3 STEM index).
        mrl_dims = int(getattr(self.settings, "embed_mrl_dims", 0) or 0) if model is None else 0
        # Cache key = (model, dims, text) — dims included so a change in embed_mrl_dims
        # never returns a stale full-width vector. Bounded FIFO eviction.
        key = f"{embed_model}:{mrl_dims}::{text}"
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        try:
            if embed_spec is not None:
                vec = await embed_one(self._client, embed_spec, text, timeout=self.settings.llm_timeout)
            else:
                r = await self._client.post(
                    f"{self.settings.ollama_host}/api/embeddings",
                    json={"model": embed_model, "prompt": text}, timeout=self.settings.llm_timeout,
                )
                r.raise_for_status()
                vec = list(r.json().get("embedding") or [])
        except httpx.HTTPError as e:
            raise OllamaError(f"embed failed: {e}") from e
        if vec and mrl_dims:
            vec = truncate_mrl(vec, mrl_dims)
        if vec:
            if len(self._embed_cache) >= self._EMBED_CACHE_MAX:
                # Evict oldest insertion.
                with contextlib.suppress(StopIteration):
                    self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[key] = vec
        return vec


_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
