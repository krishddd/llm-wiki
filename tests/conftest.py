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
        return [
            "nomic-embed-text:latest",
            "qwen3:14b",
            "gemma3:e4b",
            "llama3.2:latest",
            "llava:7b",
        ]

    async def gemma(self, prompt: str, system: str | None = None, *, temperature: float = 0.4) -> str:
        await self._track()
        try:
            await asyncio.sleep(0.01)
            self.call_log.append({"model": "gemma", "len": len(prompt)})
            if "Extract named entities" in prompt:
                return (
                    '{"entities":[{"name":"Docker","type":"CONCEPT"},'
                    '{"name":"FastAPI","type":"CONCEPT"}],'
                    '"relations":[{"src":"Docker","src_type":"CONCEPT","dst":"FastAPI","dst_type":"CONCEPT",'
                    '"rel_type":"RELATES_TO"}]}'
                )
            return "Chunk summary covering docker, fastapi, and qwen."
        finally:
            await self._release()

    async def qwen(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        self.call_log.append({"model": "qwen", "len": len(prompt)})
        if "confidence" in (system or "") or "confidence" in prompt.lower()[:200]:
            return '{"confidence": 0.82, "reason": "faithful summary"}'
        if "merging partial" in prompt:
            return "Merged summary: docker, fastapi, qwen integration for the wiki."
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

    async def llama(self, prompt: str, system: str | None = None, *, temperature: float = 0.3) -> str:
        return "llama fallback"

    async def llava(self, prompt: str, image_path) -> str:
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
