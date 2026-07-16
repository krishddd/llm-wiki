"""Backfill v5 RAG artifacts for pages ingested before the v5 package landed.

Two artifacts:
1. Doc2Query `#hq` units — for every knowledge page missing `hypothetical_questions`,
   generate the questions it answers (one summary-model call per page), persist them
   to frontmatter, and index them as `<pid>#hq` in BM25 + dense.
2. Topic overview pages — one `build_topic_pages()` run (RAPTOR-lite clustering).

Requires a live Ollama. If the configured models (gemma4/qwen3) aren't pulled yet,
point the roles at any installed chat model:

    python scripts/backfill_v5.py --summary-model llama3.2:latest --reason-model llama3.2:latest

Idempotent: pages that already carry `hypothetical_questions` are skipped, and the
topic build replaces previous topic pages.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="max pages to backfill (0 = all)")
    ap.add_argument("--skip-hq", action="store_true")
    ap.add_argument("--skip-topics", action="store_true")
    ap.add_argument("--summary-model", default="", help="override MODEL_SUMMARY (doc2query)")
    ap.add_argument("--reason-model", default="", help="override MODEL_REASON (topic summaries)")
    args = ap.parse_args()

    # Model overrides must land in the environment BEFORE Settings is constructed.
    if args.summary_model:
        os.environ["MODEL_SUMMARY"] = args.summary_model
    if args.reason_model:
        os.environ["MODEL_REASON"] = args.reason_model
        os.environ["MODEL_FAST"] = args.reason_model  # keep fallback path a no-op

    asyncio.run(_run(args))


async def _run(args) -> None:
    from src.config import get_settings
    from src.ingest import Ingestor
    from src.llm import get_client
    from src.search.bm25_index import BM25Index
    from src.search.dense_index import DenseIndex
    from src.wiki.pages import PageStore, write_page

    s = get_settings()
    client = get_client()
    bm25 = BM25Index(Path(args.data_dir) / "bm25.pkl")
    dense = DenseIndex(Path(args.data_dir) / "chroma", embed_fn=client.embed)
    store = PageStore(Path(args.wiki_dir))

    done = 0
    skipped = 0
    if not args.skip_hq:
        ingestor = Ingestor(settings=s, client=client, graph=None, bm25=bm25, dense=dense)
        t0 = time.time()
        for pid, page in store.iter_pages(("sources",)):
            fm = page.frontmatter or {}
            kind = str(fm.get("kind", "source")).lower()
            if kind in ("entity", "topic", "procedure", "episodic"):
                continue
            if fm.get("hypothetical_questions"):
                skipped += 1
                continue
            if args.limit and done >= args.limit:
                break
            title = str(fm.get("title") or Path(pid).stem)
            seed = f"{fm.get('description', '')}\n\n{page.body[:2500]}"
            questions = await ingestor._doc2query(seed)
            if not questions:
                print(f"  no questions generated for {pid} (LLM unavailable or empty)")
                continue
            fm["hypothetical_questions"] = questions
            page.frontmatter = fm
            write_page(page)
            await ingestor._index_doc2query(pid, title, questions)
            done += 1
            print(f"  [{done}] {pid} -> {len(questions)} questions "
                  f"({(time.time() - t0) / done:.1f}s/page avg)")
        print(f"hq backfill: {done} pages backfilled, {skipped} already had questions")

    if not args.skip_topics:
        from src.wiki.topics import build_topic_pages
        result = await build_topic_pages(
            wiki_dir=Path(args.wiki_dir), client=client, bm25=bm25, dense=dense,
            min_cluster=s.topics_min_cluster, max_topics=s.topics_max,
            sim_threshold=s.topics_sim_threshold,
        )
        print(f"topics: {result}")

    await client.aclose()


if __name__ == "__main__":
    main()
