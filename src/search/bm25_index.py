"""BM25 keyword index backed by rank-bm25, persisted as pickle."""
from __future__ import annotations

import asyncio
import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._docs: dict[str, list[str]] = {}
        self._bm25: BM25Okapi | None = None
        self._ordered_ids: list[str] = []
        self._lock = asyncio.Lock()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("rb") as f:
            self._docs = pickle.load(f)
        self._rebuild()

    def _rebuild(self) -> None:
        self._ordered_ids = list(self._docs.keys())
        corpus = [self._docs[pid] for pid in self._ordered_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(self._docs, f)

    async def upsert(self, page_id: str, text: str) -> None:
        async with self._lock:
            self._docs[page_id] = _tokenise(text)
            self._rebuild()
            self._persist()

    async def delete(self, page_id: str) -> None:
        async with self._lock:
            if page_id in self._docs:
                del self._docs[page_id]
                self._rebuild()
                self._persist()

    async def search(self, query: str, k: int = 20) -> list[str]:
        async with self._lock:
            if self._bm25 is None:
                return []
            tokens = _tokenise(query)
            if not tokens:
                return []
            # Only keep docs that contain at least one query token — BM25 scores can be
            # negative or zero (e.g. N=1 or token-in-every-doc) but those docs are still
            # lexical matches and should surface for downstream reranking.
            query_set = set(tokens)
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(zip(self._ordered_ids, scores), key=lambda kv: kv[1], reverse=True)
            out: list[str] = []
            for pid, _ in ranked:
                if query_set & set(self._docs[pid]):
                    out.append(pid)
                if len(out) >= k:
                    break
            return out
