"""Golden-set evaluation harness for the retrieval + synthesis pipeline.

The pipeline stacks ~16 techniques (HyDE, RAG-Fusion, CRAG, MMR, small-to-big,
Doc2Query, …) — this harness measures which of them actually pay for their latency
on YOUR corpus, instead of trusting the papers.

Golden set format (`eval/golden.jsonl`, one JSON object per line):

    {"question": "…", "expected_pages": ["sources/foo.md"], "keywords": ["…"], "domain": "general"}

- `expected_pages`: page-ids that a correct retrieval must surface.
- `keywords`: strings a correct *answer* should contain (answer eval only).

Three layers:
- `build_eval_indexes()` — self-contained BM25 + dense built fresh from the wiki
  (same sub-chunk scheme as ingest), so results reflect current content and the
  harness runs without the app's Docker state.
- `eval_retrieval()` — recall@k, MRR, hit-rate, latency per variant. No synthesis
  LLM needed (only the embedding model), so ablations are cheap to run.
- `eval_answers()` — full QueryEngine run: keyword coverage, groundedness,
  confidence calibration, reasoner used, latency. Needs the chat models.

`RETRIEVAL_ABLATIONS` defines one-flag-off variants for `scripts/run_eval.py --ablate`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .search.hybrid import hybrid_search

log = logging.getLogger(__name__)


@dataclass
class GoldenItem:
    question: str
    expected_pages: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    domain: str = "general"


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        items.append(GoldenItem(
            question=str(d["question"]),
            expected_pages=[str(p).replace("\\", "/") for p in d.get("expected_pages", [])],
            keywords=[str(k) for k in d.get("keywords", [])],
            domain=str(d.get("domain", "general")),
        ))
    return items


def save_golden(path: Path, items: list[GoldenItem]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "question": it.question,
                "expected_pages": it.expected_pages,
                "keywords": it.keywords,
                "domain": it.domain,
            }, ensure_ascii=False) + "\n")


async def build_eval_indexes(wiki_dir: Path, client, workdir: Path):
    """Fresh BM25 + dense indexes over the live wiki, mirroring ingest's sub-chunk
    scheme (including `#hq` units from frontmatter). Returns (bm25, dense, page_store)."""
    from .config import get_settings
    from .ingest import Ingestor
    from .search.bm25_index import BM25Index
    from .search.dense_index import DenseIndex
    from .wiki.pages import PageStore

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    bm25_path = workdir / "eval_bm25.pkl"
    if bm25_path.exists():
        bm25_path.unlink()
    bm25 = BM25Index(bm25_path)
    dense = DenseIndex(workdir / "eval_chroma", embed_fn=client.embed, collection="eval_pages")
    store = PageStore(wiki_dir)

    ingestor = Ingestor(settings=get_settings(), client=client, graph=None, bm25=bm25, dense=dense)
    n = 0
    for pid, page in store.iter_pages(("sources", "entities")):
        fm = page.frontmatter or {}
        title = str(fm.get("title") or Path(pid).stem)
        await ingestor._index_page_chunks(pid, title, page.body, fm)
        hq = fm.get("hypothetical_questions") or []
        if hq:
            await ingestor._index_doc2query(pid, title, [str(q) for q in hq])
        n += 1
    log.info("eval indexes built", extra={"metadata": {"pages": n, "dense_backend": dense.backend}})
    return bm25, dense, store


# One-flag-off retrieval variants (kwargs passed to hybrid_search).
RETRIEVAL_ABLATIONS: dict[str, dict] = {
    "baseline": {},
    "no_chunk_context": {"use_chunk_context": False},
    "no_mmr": {"use_mmr": False},
    "no_synth_downweight": {"synth_downweight": 1.0},
    "no_graph_expand": {"graph_expand": False},
}


async def eval_retrieval(
    golden: list[GoldenItem],
    *,
    bm25,
    dense,
    page_store,
    graph=None,
    top_k: int = 5,
    **hybrid_kwargs,
) -> dict:
    """Recall@k / MRR / hit-rate / latency over the golden set for one variant."""
    per_item: list[dict] = []
    for item in golden:
        t0 = time.perf_counter()
        try:
            retrieved = await hybrid_search(
                item.question, bm25=bm25, dense=dense, page_store=page_store,
                graph=graph, top_k_rerank=top_k, **hybrid_kwargs,
            )
        except Exception as e:
            log.warning("eval retrieval failed", extra={"metadata": {"q": item.question[:60], "error": str(e)[:160]}})
            retrieved = []
        latency_ms = (time.perf_counter() - t0) * 1000
        got = [r.page_id for r in retrieved]
        expected = set(item.expected_pages)
        found = expected & set(got)
        recall = len(found) / len(expected) if expected else 0.0
        rr = 0.0
        for rank, pid in enumerate(got, start=1):
            if pid in expected:
                rr = 1.0 / rank
                break
        per_item.append({
            "question": item.question,
            "expected": sorted(expected),
            "retrieved": got,
            "recall": round(recall, 3),
            "reciprocal_rank": round(rr, 3),
            "latency_ms": round(latency_ms, 1),
        })

    n = len(per_item) or 1
    return {
        "items": len(per_item),
        "top_k": top_k,
        f"recall_at_{top_k}": round(sum(i["recall"] for i in per_item) / n, 3),
        "mrr": round(sum(i["reciprocal_rank"] for i in per_item) / n, 3),
        "hit_rate": round(sum(1 for i in per_item if i["reciprocal_rank"] > 0) / n, 3),
        "avg_latency_ms": round(sum(i["latency_ms"] for i in per_item) / n, 1),
        "per_item": per_item,
    }


async def eval_answers(golden: list[GoldenItem], engine, *, top_k: int = 5) -> dict:
    """Full-pipeline answer quality: keyword coverage, groundedness, confidence."""
    per_item: list[dict] = []
    for item in golden:
        t0 = time.perf_counter()
        try:
            result = await engine.answer(item.question, top_k=top_k, save_back=False)
        except Exception as e:
            log.warning("eval answer failed", extra={"metadata": {"q": item.question[:60], "error": str(e)[:160]}})
            continue
        latency_s = time.perf_counter() - t0
        answer_low = (result.answer or "").lower()
        kw_hits = [k for k in item.keywords if k.lower() in answer_low]
        coverage = len(kw_hits) / len(item.keywords) if item.keywords else 1.0
        cited = set(result.retrieved_pages) & set(item.expected_pages)
        per_item.append({
            "question": item.question,
            "keyword_coverage": round(coverage, 3),
            "grounded": result.grounded,
            "confidence": result.confidence,
            "expected_page_retrieved": bool(cited) if item.expected_pages else None,
            "reasoner": result.reasoner,
            "latency_s": round(latency_s, 1),
        })

    n = len(per_item) or 1
    return {
        "items": len(per_item),
        "avg_keyword_coverage": round(sum(i["keyword_coverage"] for i in per_item) / n, 3),
        "grounded_rate": round(sum(1 for i in per_item if i["grounded"]) / n, 3),
        "avg_confidence": round(sum(i["confidence"] for i in per_item) / n, 3),
        "avg_latency_s": round(sum(i["latency_s"] for i in per_item) / n, 1),
        "per_item": per_item,
    }
