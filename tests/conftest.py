from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings


class FakeOllama:
    """In-memory fake Ollama client for unit tests."""

    def __init__(self):
        self.call_log: list[dict[str, Any]] = []
        self.max_concurrent_seen = 0
        self._live = 0
        self._lock = asyncio.Lock()

    async def _track(self):
        async with self._lock:
            self._live += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._live)

    async def _release(self):
        async with self._lock:
            self._live -= 1

    async def list_models(self) -> list[str]:
        # Mirror the default 4-model stack in config.required_models() so /health
        # reports models_missing == [] under the fake client.
        return [
            "nomic-embed-text:latest",
            "qwen3:14b",
            "gemma4:e4b",
            "llava:7b",
        ]

    async def summarize(self, prompt: str, system: str | None = None, *, temperature: float = 0.4) -> str:
        await self._track()
        try:
            await asyncio.sleep(0.01)
            self.call_log.append({"role": "summary", "len": len(prompt)})
            if "questions that this document" in prompt:
                return (
                    '{"questions":["What is Docker used for?","How does FastAPI '
                    'serve the wiki?","Which model powers reasoning?"]}'
                )
            if "Extract named entities" in prompt:
                return (
                    '{"entities":[{"name":"Docker","type":"CONCEPT"},'
                    '{"name":"FastAPI","type":"CONCEPT"}],'
                    '"relations":[{"src":"Docker","src_type":"CONCEPT","dst":"FastAPI","dst_type":"CONCEPT",'
                    '"rel_type":"RELATES_TO"}]}'
                )
            return (
                "Chunk summary covering docker, fastapi, and qwen. "
                "The document explains how Docker containers host the FastAPI service "
                "and how the qwen model integrates with the LLM wiki pipeline."
            )
        finally:
            await self._release()

    async def reason(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        self.call_log.append({"role": "reason", "len": len(prompt)})
        if "confidence" in (system or "") or "confidence" in prompt.lower()[:200]:
            return '{"confidence": 0.82, "reason": "faithful summary"}'
        if "merging partial" in prompt:
            return (
                "Merged summary: docker, fastapi, qwen integration for the wiki. "
                "Docker containers run the FastAPI application while the qwen model "
                "provides reasoning and synthesis for the knowledge pipeline."
            )
        if '"answer"' in (system or ""):
            return (
                '{"answer":"Docker and FastAPI are discussed [Sample Page].",'
                '"summary":"Docker + FastAPI notes.",'
                '"key_points":["Docker runs the app","FastAPI exposes endpoints"],'
                '"entities":["Docker","FastAPI"],"confidence":0.8}'
            )
        if '"orphans"' in (system or ""):
            return '{"orphans":[],"stale":[],"missing_entity_pages":[],"contradictions":[],"suggested_sources":[]}'
        return "qwen default response"

    async def fast(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        return "llama fallback"

    async def vision(self, prompt: str, image_path) -> str:
        return "image described"

    async def embed(self, text: str) -> list[float]:
        v = [0.0] * 16
        for i, ch in enumerate(text.encode()[:16]):
            v[i] = ch / 255.0
        return v

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    (tmp_path / "wiki").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    for sub in ("sources", "entities", "review", "raw"):
        (tmp_path / "wiki" / sub).mkdir(parents=True, exist_ok=True)
    return Settings(
        ollama_host="http://fake",
        wiki_dir=tmp_path / "wiki",
        raw_dir=tmp_path / "wiki" / "raw",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        max_concurrent_llm_req=2,
        confidence_threshold=0.60,
    )
