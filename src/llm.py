"""Async Ollama client. Per-role helpers + timeout-triggered fallback to model_fast.

All calls hit the remote `OLLAMA_HOST` via `/api/chat` and `/api/embeddings`.
If `model_fast == model_reason`, the fallback is skipped and the original error propagates.
"""
from __future__ import annotations

import base64
import contextlib
import logging
import re
from pathlib import Path

import httpx

from .config import Settings, get_settings

log = logging.getLogger(__name__)

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

        Hosted providers (Groq / GitHub Models / Gemini) are OpenAI-compatible; a
        provider HTTP error is re-raised as OllamaError so existing role fallbacks
        (e.g. qwen→llama) still apply. Missing key/model → transparent Ollama path.
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

    async def gemma(self, prompt: str, system: str | None = None, *, temperature: float = 0.4) -> str:
        return await self._role_chat("summary", self.settings.model_summary, prompt, system, temperature=temperature)

    async def qwen(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        try:
            return await self._role_chat("reason", self.settings.model_reason, prompt, system, temperature=temperature)
        except OllamaError as e:
            # Only fall back if model_fast is a genuinely different model; otherwise re-raise
            # so the caller's own fallback (e.g. ingest's merge-concat) kicks in instead of
            # retrying the same overloaded model.
            if self.settings.model_fast and self.settings.model_fast != self.settings.model_reason:
                log.warning(
                    f"qwen failed, falling back to {self.settings.model_fast}",
                    extra={"metadata": {"error": str(e)}},
                )
                return await self.llama(prompt, system, temperature=temperature)
            raise

    async def llama(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        # Named "llama" for historical reasons — actually dispatches to the fast role
        # (model_fast on Ollama, or the provider_fast hosted model, e.g. Groq).
        return await self._role_chat(
            "fast", self.settings.model_fast, prompt, system,
            temperature=temperature, timeout=self.settings.llm_fast_timeout,
        )

    async def solver(self, prompt: str, system: str | None = None, *, temperature: float | None = None) -> str:
        """Reasoning specialist (VibeThinker). Returns the answer with any
        <think> trace stripped. Caller is responsible for routing only quantitative
        tasks here — VibeThinker is weak on broad-knowledge / recall.

        Raises OllamaError if `model_solver` is unset or the model isn't served;
        callers should catch and fall back to qwen synthesis.
        """
        spec_provider = self.settings.provider_solver and self.settings.provider_solver.lower() != "ollama"
        if not self.settings.model_solver and not spec_provider:
            raise OllamaError("model_solver is not configured")
        temp = self.settings.solver_temperature if temperature is None else temperature
        raw = await self._role_chat("solver", self.settings.model_solver, prompt, system, temperature=temp)
        return strip_think(raw)

    async def llava(self, prompt: str, image_path: str | Path) -> str:
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        return await self._chat(self.settings.model_vision, prompt, None, images=[img_b64])

    async def embed(self, text: str, *, model: str | None = None) -> list[float]:
        # A `model` override (used by the STEM-routed dense index) always means a
        # specific Ollama embedder; only the default path may route to a provider.
        from .providers import embed_one, resolve_embed_provider
        embed_spec = resolve_embed_provider(self.settings) if model is None else None
        embed_model = model or (embed_spec.model if embed_spec else self.settings.model_embed)
        # Cache key = (model, text) — bounded FIFO eviction.
        key = f"{embed_model}::{text}"
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
