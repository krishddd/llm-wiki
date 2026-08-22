"""Rebuild the BM25 + dense retrieval indexes from the on-disk wiki (source of truth).

The Markdown pages under `wiki/` are authoritative; the vectors in `data/chroma`
and `data/bm25.pkl` are always re-derivable by re-embedding them. Run this whenever
the embedding space changes and stored vectors would no longer match query vectors —
most importantly after changing `EMBED_MRL_DIMS` (Matryoshka truncation), which
changes the dense vector dimensionality (e.g. 768 → 256). Mixing 256-dim queries
with 768-dim stored vectors silently returns zero matches, so a full re-embed is
mandatory when that knob changes.

What it does, per page under sources/ + entities/ + procedures/:
  - re-index the small-to-big sub-chunks (`<pid>#<n>`) into BM25 + dense, using the
    SAME `index_page_chunks` primitive ingest uses (contextual preamble + dense meta),
  - stamp the fresh `chunk_count` back into frontmatter so later deletes are exact.

It first WIPES `data/chroma` (and `chroma_stem`) + `data/bm25.pkl` so no stale
dimensionality or orphan units survive. Doc2Query `#hq` units and media `#media#<n>`
embeddings are NOT regenerated here (they need an LLM / the graph) — run a full
re-ingest, or `scripts/backfill_v5.py`, to restore `#hq` units.

Requires a live Ollama with the embedder pulled. Override the embed model if needed:

    python scripts/rebuild_index.py --embed-model nomic-embed-text:latest
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="max pages to index (0 = all)")
    ap.add_argument("--embed-model", default="", help="override MODEL_EMBED for this run")
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT wipe the indexes first (risky — leaves stale-dim vectors)")
    args = ap.parse_args()

    if args.embed_model:
        os.environ["MODEL_EMBED"] = args.embed_model

    asyncio.run(_run(args))


async def _run(args) -> None:
    from llm_wiki.config import get_settings
    from llm_wiki.llm import get_client
    from llm_wiki.search.bm25_index import BM25Index
    from llm_wiki.search.dense_index import DenseIndex
    from llm_wiki.wiki.pages import PageStore, write_page
    from llm_wiki.wiki.reindex import index_page_chunks

    s = get_settings()
    data_dir = Path(args.data_dir)
    mrl = int(getattr(s, "embed_mrl_dims", 0) or 0)
    print(f"embedder={s.model_embed}  mrl_dims={mrl or 'off (full-width)'}")

    if not args.keep_existing:
        for name in ("chroma", "chroma_stem"):
            p = data_dir / name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                print(f"wiped {p}")
        bm = data_dir / "bm25.pkl"
        if bm.exists():
            bm.unlink()
            print(f"wiped {bm}")

    client = get_client()
    bm25 = BM25Index(data_dir / "bm25.pkl")
    dense = DenseIndex(data_dir / "chroma", embed_fn=client.embed)
    store = PageStore(Path(args.wiki_dir))

    t0 = time.time()
    n_pages = n_chunks = 0
    for pid, page in store.iter_pages():
        title = page.frontmatter.get("title") or pid
        count = await index_page_chunks(bm25, dense, pid, title, page.body, page.frontmatter)
        # Keep frontmatter's chunk_count truthful so a later delete/reindex is exact.
        if page.frontmatter.get("chunk_count") != count:
            page.frontmatter["chunk_count"] = count
            write_page(page)
        n_pages += 1
        n_chunks += count
        if n_pages % 25 == 0:
            print(f"  … {n_pages} pages, {n_chunks} chunks")
        if args.limit and n_pages >= args.limit:
            break

    print(f"done: {n_pages} pages → {n_chunks} chunks in {time.time() - t0:.1f}s "
          f"(dense backend: {dense.backend})")


if __name__ == "__main__":
    main()
