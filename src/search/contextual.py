"""Contextual Retrieval (Anthropic, 2024).

For each chunk, generate a short 1-2 sentence contextual preamble situating the
chunk in the document. Prepend it to the chunk text BEFORE embedding / BM25
indexing — but keep the original chunk for display.

Result: chunk vectors carry "what doc / what section / what topic" signal that
plain chunk text lacks, which improves recall on queries phrased differently
than the source.

Anthropic measured -35% top-20 retrieval failure with contextual embeddings
alone; -67% combined with BM25 + reranking. Our pipeline already has BM25 +
rerank + graph + MMR, so this is the missing layer.
"""
from __future__ import annotations

import asyncio
import logging

from ..llm import OllamaClient

log = logging.getLogger(__name__)

CONTEXTUALIZE_SYSTEM = (
    "You write a short context preamble for a chunk of text from a longer document. "
    "Goal: help future search find this chunk by name, topic, or relation. "
    "In ONE sentence (max 30 words), state: what document this is from, what section/topic "
    "this chunk covers, and any key entities. No fluff. Reply ONLY with the sentence."
)


def _build_prompt(doc_title: str, full_doc_excerpt: str, chunk: str) -> str:
    return (
        f"DOCUMENT TITLE: {doc_title}\n\n"
        f"DOCUMENT (truncated):\n{full_doc_excerpt[:4000]}\n\n"
        f"CHUNK to situate:\n{chunk[:2000]}\n\n"
        "Context preamble:"
    )


async def contextualize_chunk(
    client: OllamaClient,
    *,
    doc_title: str,
    full_doc_excerpt: str,
    chunk: str,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Return a short context preamble. Falls back to a deterministic "From <doc>:" stub on failure."""
    sem = semaphore or asyncio.Semaphore(1)
    fallback = f"From '{doc_title}':"
    try:
        async with sem:
            text = await client.gemma(
                _build_prompt(doc_title, full_doc_excerpt, chunk),
                system=CONTEXTUALIZE_SYSTEM,
                temperature=0.2,
            )
        text = (text or "").strip().split("\n")[0].strip()
        if not text or len(text) > 400:
            return fallback
        return text
    except Exception as e:
        log.debug("contextualize failed", extra={"metadata": {"error": str(e)[:120]}})
        return fallback


async def contextualize_chunks(
    client: OllamaClient,
    *,
    doc_title: str,
    full_text: str,
    chunks: list[str],
    semaphore: asyncio.Semaphore | None = None,
) -> list[str]:
    """Bulk variant — produces one preamble per chunk in parallel (bounded by semaphore)."""
    if not chunks:
        return []
    excerpt = full_text[:4000]
    tasks = [
        contextualize_chunk(
            client, doc_title=doc_title, full_doc_excerpt=excerpt, chunk=c, semaphore=semaphore
        )
        for c in chunks
    ]
    return await asyncio.gather(*tasks)


def merge_context_with_chunk(context: str, chunk: str) -> str:
    """The string that gets embedded / BM25-indexed. The chunk itself stays untouched
    for display; this is a parallel "search vector text" only."""
    if not context:
        return chunk
    return f"[Context] {context}\n\n{chunk}"
