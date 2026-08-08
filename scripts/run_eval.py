"""Run the golden-set evaluation.

Retrieval eval (default) needs only the embedding model — cheap enough to run per
ablation variant. Answer eval (--answers) runs the full QueryEngine per question
(chat models required — expect minutes per question on local CPU models).

    python scripts/run_eval.py                 # retrieval baseline
    python scripts/run_eval.py --ablate        # baseline + one-flag-off variants
    python scripts/run_eval.py --answers       # + full answer-quality pass

Results land in eval/results-<label>.json; a summary table prints to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--golden", default=Path("eval/golden.jsonl"), type=Path)
    ap.add_argument("--workdir", default=Path("eval/.work"), type=Path)
    ap.add_argument("--out-dir", default=Path("eval"), type=Path)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ablate", action="store_true", help="run one-flag-off retrieval variants")
    ap.add_argument("--answers", action="store_true", help="also run full answer-quality eval")
    ap.add_argument("--label", default="", help="label for the results file")
    ap.add_argument("--summary-model", default="", help="override MODEL_SUMMARY")
    ap.add_argument("--reason-model", default="", help="override MODEL_REASON")
    args = ap.parse_args()
    if args.summary_model:
        os.environ["MODEL_SUMMARY"] = args.summary_model
    if args.reason_model:
        os.environ["MODEL_REASON"] = args.reason_model
        os.environ["MODEL_FAST"] = args.reason_model
    asyncio.run(_run(args))


async def _run(args) -> None:
    from llm_wiki.eval_harness import (
        RETRIEVAL_ABLATIONS,
        build_eval_indexes,
        eval_answers,
        eval_retrieval,
        load_golden,
    )
    from llm_wiki.llm import get_client

    if not args.golden.exists():
        print(f"golden set not found: {args.golden} — run scripts/gen_golden.py first")
        return
    golden = load_golden(args.golden)
    print(f"golden set: {len(golden)} items")

    client = get_client()
    t0 = time.time()
    bm25, dense, store = await build_eval_indexes(args.wiki_dir, client, args.workdir)
    print(f"indexes built in {time.time() - t0:.0f}s (dense backend: {dense.backend})")

    variants = RETRIEVAL_ABLATIONS if args.ablate else {"baseline": {}}
    results: dict = {"golden": str(args.golden), "items": len(golden), "retrieval": {}}
    for name, kwargs in variants.items():
        r = await eval_retrieval(
            golden, bm25=bm25, dense=dense, page_store=store, top_k=args.top_k, **kwargs,
        )
        results["retrieval"][name] = r
        print(f"  {name:22s} recall@{args.top_k}={r[f'recall_at_{args.top_k}']:.3f} "
              f"mrr={r['mrr']:.3f} hit={r['hit_rate']:.3f} {r['avg_latency_ms']:.0f}ms")

    if args.answers:
        from llm_wiki.config import get_settings
        from llm_wiki.query import QueryEngine
        engine = QueryEngine(bm25=bm25, dense=dense, page_store=store,
                             graph=None, settings=get_settings(), client=client)
        engine.procedures = None
        print("running answer eval (full pipeline — this is the slow part)…")
        a = await eval_answers(golden, engine, top_k=args.top_k)
        results["answers"] = a
        print(f"  answers: keyword_coverage={a['avg_keyword_coverage']:.3f} "
              f"grounded={a['grounded_rate']:.3f} conf={a['avg_confidence']:.3f} "
              f"{a['avg_latency_s']:.0f}s/q")

    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"results-{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults -> {out}")
    await client.aclose()


if __name__ == "__main__":
    main()
