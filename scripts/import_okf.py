"""Import an external OKF bundle into the wiki (curated — no LLM pipeline).

    python scripts/import_okf.py path/to/bundle [--no-graph] [--no-index]

Pages land in wiki/sources/ as okf-<bundle>-<slug>.md with provenance stamped;
markdown links become RELATES_TO edges; chunks are indexed into BM25 + dense.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    asyncio.run(_run(args))


async def _run(args) -> None:
    from src.loaders.okf_loader import import_okf_bundle, validate_bundle

    report = validate_bundle(args.bundle)
    print(f"bundle validation: conformant={report['conformant']} pages={report['pages']}")
    for v in report["violations"][:10]:
        print(f"  violation: {v}")
    if args.validate_only:
        return

    bm25 = dense = graph = client = None
    if not args.no_index:
        from src.llm import get_client
        from src.search.bm25_index import BM25Index
        from src.search.dense_index import DenseIndex
        client = get_client()
        bm25 = BM25Index(Path(args.data_dir) / "bm25.pkl")
        dense = DenseIndex(Path(args.data_dir) / "chroma", embed_fn=client.embed)
    if not args.no_graph:
        from src.graph import KnowledgeGraph
        graph = KnowledgeGraph(Path(args.data_dir) / "graph.db")

    result = await import_okf_bundle(
        args.bundle, wiki_dir=args.wiki_dir, bm25=bm25, dense=dense, graph=graph, client=client,
    )
    print(f"imported: {result.pages_imported} pages, {result.relations_added} relations, "
          f"{result.pages_skipped} skipped, {len(result.errors)} errors")
    for e in result.errors[:10]:
        print(f"  error: {e}")

    from src.wiki.index_md import rebuild_index
    rebuild_index(Path(args.wiki_dir))
    if client is not None:
        await client.aclose()


if __name__ == "__main__":
    main()
