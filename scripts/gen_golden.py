"""Generate a golden evaluation set from the live wiki.

Samples N knowledge pages (deterministic seed) and asks the summary model, for
each, one question that the page uniquely answers plus 2-4 answer keywords.
Writes `eval/golden.jsonl` for `scripts/run_eval.py`.

Model override (when gemma4 isn't pulled):

    python scripts/gen_golden.py --n 15 --summary-model llama3.2:latest

Review the generated questions before trusting eval numbers — LLM-generated golden
sets inherit the generator's blind spots; hand-edit/extend the JSONL freely.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_GOLDEN_SYSTEM = (
    "You write evaluation questions for a retrieval system. Given a wiki page's "
    "title and content, produce ONE specific question that THIS page (and ideally "
    "only this page) answers, plus 2-4 short keywords/phrases a correct answer "
    "must contain. The question must not quote the title verbatim.\n"
    'Reply ONLY JSON: {"question":"…","keywords":["…","…"]}'
)


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--out", default=Path("eval/golden.jsonl"), type=Path)
    ap.add_argument("--n", type=int, default=15, help="pages to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="append to an existing golden file")
    ap.add_argument("--summary-model", default="", help="override MODEL_SUMMARY")
    args = ap.parse_args()
    if args.summary_model:
        os.environ["MODEL_SUMMARY"] = args.summary_model
    asyncio.run(_run(args))


async def _run(args) -> None:
    from llm_wiki.eval_harness import GoldenItem, load_golden, save_golden
    from llm_wiki.llm import get_client
    from llm_wiki.wiki.pages import PageStore

    client = get_client()
    store = PageStore(Path(args.wiki_dir))

    candidates: list[tuple[str, str, str, str]] = []  # (pid, title, domain, content)
    for pid, page in store.iter_pages(("sources",)):
        fm = page.frontmatter or {}
        if str(fm.get("kind", "source")).lower() not in ("source", "synthesis", "promoted"):
            continue
        title = str(fm.get("title") or Path(pid).stem)
        content = f"{fm.get('description', '')}\n\n{page.body[:2000]}"
        candidates.append((pid, title, str(fm.get("domain", "general")), content))

    random.Random(args.seed).shuffle(candidates)
    sample = candidates[: args.n]
    print(f"sampling {len(sample)} of {len(candidates)} pages")

    items: list[GoldenItem] = list(load_golden(args.out)) if (args.append and args.out.exists()) else []
    for i, (pid, title, domain, content) in enumerate(sample, start=1):
        prompt = f"PAGE TITLE: {title}\n\nPAGE CONTENT:\n{content}"
        try:
            raw = await client.summarize(prompt, system=_GOLDEN_SYSTEM, temperature=0.4)
        except Exception as e:
            print(f"  [{i}] {pid}: LLM call failed ({str(e)[:80]})")
            continue
        data = _extract_json(raw) or {}
        q = str(data.get("question") or "").strip()
        kws = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()]
        if len(q) < 12:
            print(f"  [{i}] {pid}: no usable question")
            continue
        items.append(GoldenItem(question=q, expected_pages=[pid], keywords=kws[:4], domain=domain))
        print(f"  [{i}] {pid}: {q[:80]}")

    save_golden(args.out, items)
    print(f"\nwrote {len(items)} golden items -> {args.out}")
    await client.aclose()


if __name__ == "__main__":
    main()
