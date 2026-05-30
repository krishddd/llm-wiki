"""Dense vector index.

Tries ChromaDB (persistent → ephemeral fallback). If chromadb itself is broken on
the host (e.g. a version bug where `get_or_create_collection` raises
`KeyError: '_type'` during collection config migration), falls back to a pure
in-process numpy-based cosine index. The wiki Markdown files on disk are the
source of truth — vectors are always rebuildable by re-embedding pages.
"""
from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

log = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[list[float]]]


class _NumpyCosineIndex:
    """In-memory cosine-similarity index. No external deps, no on-disk state."""

    def __init__(self) -> None:
        self._vecs: dict[str, list[float]] = {}
        self._docs: dict[str, str] = {}
        self._meta: dict[str, dict] = {}

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)

    def upsert(self, page_id: str, vec: list[float], text: str, meta: dict) -> None:
        self._vecs[page_id] = list(vec)
        self._docs[page_id] = text
        self._meta[page_id] = dict(meta)

    def delete(self, page_id: str) -> None:
        self._vecs.pop(page_id, None)
        self._docs.pop(page_id, None)
        self._meta.pop(page_id, None)

    def query(self, vec: list[float], k: int) -> list[str]:
        if not self._vecs:
            return []
        scored = [(pid, self._cos(vec, v)) for pid, v in self._vecs.items()]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [pid for pid, _ in scored[:k]]


class DenseIndex:
    def __init__(self, persist_dir: Path, embed_fn: EmbedFn, collection: str = "wiki_pages"):
        self.path = Path(persist_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._embed = embed_fn
        self._chroma = None
        self._collection = None
        self._np_fallback: _NumpyCosineIndex | None = None
        self._try_chroma(collection)

    def _try_chroma(self, collection: str) -> None:
        try:
            import chromadb
        except Exception as e:
            log.warning("chromadb not importable, using numpy fallback", extra={"metadata": {"error": str(e)}})
            self._np_fallback = _NumpyCosineIndex()
            return

        # 1) Try PersistentClient.
        try:
            self._chroma = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._chroma.get_or_create_collection(
                name=collection, metadata={"hnsw:space": "cosine"}
            )
            return
        except Exception as e:
            msg = str(e)
            log.warning("chroma persistent failed, wiping + retrying", extra={"metadata": {"error": msg}})

        # 2) Wipe persisted state and retry PersistentClient once.
        try:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._chroma.get_or_create_collection(
                name=collection, metadata={"hnsw:space": "cosine"}
            )
            return
        except Exception as e:
            log.warning("chroma persistent still failing, trying ephemeral", extra={"metadata": {"error": str(e)}})

        # 3) Try EphemeralClient (in-memory, no disk).
        try:
            self._chroma = chromadb.EphemeralClient()
            self._collection = self._chroma.get_or_create_collection(
                name=collection, metadata={"hnsw:space": "cosine"}
            )
            return
        except Exception as e:
            log.warning(
                "chromadb ephemeral also broken, using numpy fallback",
                extra={"metadata": {"error": str(e)}},
            )

        # 4) Pure-Python numpy-like cosine index.
        self._np_fallback = _NumpyCosineIndex()

    async def upsert(self, page_id: str, text: str, meta: dict | None = None) -> None:
        text = text[:8000]
        vec = await self._embed(text)
        if self._collection is not None:
            self._collection.upsert(ids=[page_id], embeddings=[vec], documents=[text], metadatas=[meta or {}])
        else:
            assert self._np_fallback is not None
            self._np_fallback.upsert(page_id, vec, text, meta or {})

    async def delete(self, page_id: str) -> None:
        if self._collection is not None:
            self._collection.delete(ids=[page_id])
        else:
            assert self._np_fallback is not None
            self._np_fallback.delete(page_id)

    async def search(self, query: str, k: int = 20) -> list[str]:
        vec = await self._embed(query)
        return await self.search_with_vec(vec, k)

    async def search_with_vec(self, vec: list[float], k: int = 20) -> list[str]:
        """Search using a precomputed embedding vector (used by HyDE)."""
        if not vec:
            return []
        if self._collection is not None:
            try:
                res = self._collection.query(query_embeddings=[vec], n_results=k)
            except Exception as e:
                log.warning("chroma query_with_vec failed", extra={"metadata": {"error": str(e)[:160]}})
                return []
            ids = (res.get("ids") or [[]])[0]
            return list(ids)
        assert self._np_fallback is not None
        return self._np_fallback.query(vec, k)

    @property
    def backend(self) -> str:
        if self._collection is not None:
            return "chromadb"
        return "numpy"
