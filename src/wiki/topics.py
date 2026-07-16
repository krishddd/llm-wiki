"""RAPTOR-lite topic pages — hierarchical corpus summaries for "global" questions.

RAPTOR (Sarthi et al. 2024) and GraphRAG's community summaries both answer the same
failure mode: chunk-level retrieval can't answer corpus-level questions ("what are the
main themes across my documents?") because no single chunk contains the answer.

This module builds one summary tier above the pages:

1. Embed every live knowledge page (title + description) — embeddings are cached
   by the LLM client, so re-runs are cheap.
2. Greedy cosine clustering (no sklearn dependency): a page joins the first cluster
   whose centroid similarity ≥ `sim_threshold`, else it seeds a new cluster.
3. Every cluster with ≥ `min_cluster` pages is summarised by the reasoning model
   into a `topic-*.md` page (kind: topic, OKF type "Topic Overview") that cites its
   member pages with bundle-relative links.

Topic pages are indexed like any other page, so a broad question retrieves the topic
overview (which the synthesiser can cite) while narrow questions still hit the leaves.
Rebuilds are idempotent: existing topic pages are removed and re-derived each run.
Scheduled weekly (`build_topics` job) and runnable via POST /admin/run/build_topics.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import re
from pathlib import Path

from .pages import PageStore
from .synth_page import SynthesisPageInputs, write_synthesis_page

log = logging.getLogger(__name__)

_TOPIC_SYSTEM = (
    "You are a knowledge-base curator. Given a cluster of related wiki pages "
    "(titles + one-line descriptions), write a topic overview.\n"
    "Reply ONLY JSON:\n"
    '{"title":"<3-6 word topic name>",'
    '"summary":"<250-400 word synthesis of what these pages collectively cover, '
    'the throughlines connecting them, and any tensions between them>",'
    '"key_themes":["…","…","…"]}'
)

_EXCLUDED_KINDS = {"topic", "entity", "episodic", "procedure"}


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)


def _mean(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


def greedy_cluster(
    items: list[tuple[str, list[float]]], *, sim_threshold: float = 0.62
) -> list[list[str]]:
    """Single-pass centroid clustering: [(id, vec), …] → [[ids…], …]."""
    centroids: list[list[float]] = []
    members: list[list[str]] = []
    vecs_by_cluster: list[list[list[float]]] = []
    for pid, vec in items:
        best_i, best_sim = -1, 0.0
        for i, cen in enumerate(centroids):
            sim = _cos(vec, cen)
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_i >= 0 and best_sim >= sim_threshold:
            members[best_i].append(pid)
            vecs_by_cluster[best_i].append(vec)
            centroids[best_i] = _mean(vecs_by_cluster[best_i])
        else:
            members.append([pid])
            vecs_by_cluster.append([vec])
            centroids.append(list(vec))
    return members


async def _remove_existing_topics(wiki_dir: Path, bm25, dense) -> int:
    """Idempotency: drop previous topic pages + their index entries before rebuild."""
    removed = 0
    for p in sorted((Path(wiki_dir) / "sources").glob("topic-*.md")):
        pid = f"sources/{p.name}"
        try:
            p.unlink()
            removed += 1
        except OSError as e:
            log.debug("topic page unlink failed", extra={"metadata": {"error": str(e)[:120]}})
            continue
        for index in (bm25, dense):
            if index is None:
                continue
            with contextlib.suppress(Exception):
                await index.delete(pid)
    return removed


async def build_topic_pages(
    *,
    wiki_dir: Path,
    client,
    bm25=None,
    dense=None,
    min_cluster: int = 3,
    max_topics: int = 12,
    sim_threshold: float = 0.62,
) -> dict:
    """Cluster live knowledge pages and write one topic-overview page per cluster."""
    wiki_dir = Path(wiki_dir)
    store = PageStore(wiki_dir)

    # 1) Collect + embed live knowledge pages.
    embedded: list[tuple[str, list[float]]] = []
    titles: dict[str, str] = {}
    descs: dict[str, str] = {}
    for pid, page in store.iter_pages(("sources",)):
        fm = page.frontmatter or {}
        kind = str(fm.get("kind", "source")).lower()
        if kind in _EXCLUDED_KINDS:
            continue
        title = str(fm.get("title") or Path(pid).stem)
        desc = str(fm.get("description") or "") or page.body[:400]
        titles[pid], descs[pid] = title, desc
        try:
            vec = await client.embed(f"{title}\n{desc}")
        except Exception as e:
            log.debug("topic embed failed", extra={"metadata": {"page": pid, "error": str(e)[:120]}})
            continue
        if vec:
            embedded.append((pid, vec))

    if len(embedded) < min_cluster:
        return {"pages_scanned": len(embedded), "clusters": 0, "topics_written": 0}

    # 2) Cluster; keep the largest clusters first.
    clusters = [c for c in greedy_cluster(embedded, sim_threshold=sim_threshold) if len(c) >= min_cluster]
    clusters.sort(key=len, reverse=True)
    clusters = clusters[:max_topics]

    removed = await _remove_existing_topics(wiki_dir, bm25, dense)

    # 3) Summarise each cluster into a topic page.
    written: list[str] = []
    for cluster in clusters:
        listing = "\n".join(f"- {titles[pid]}: {descs[pid][:200]}" for pid in cluster)
        try:
            raw = await client.qwen(f"PAGES IN CLUSTER:\n{listing}", system=_TOPIC_SYSTEM, temperature=0.3)
        except Exception as e:
            log.warning("topic summarise failed", extra={"metadata": {"error": str(e)[:160]}})
            continue
        data = _extract_json(raw) or {}
        title = str(data.get("title") or "").strip()[:120]
        summary = str(data.get("summary") or "").strip()
        if not title or len(summary) < 100:
            continue
        themes = [str(t) for t in (data.get("key_themes") or [])][:8]

        body_lines = [f"# {title}", "", summary, ""]
        if themes:
            body_lines += ["## Key themes", ""] + [f"- {t}" for t in themes] + [""]
        body_lines += ["## Member pages", ""]
        for pid in cluster:
            body_lines.append(f"- [{titles[pid]}](/{pid})")
        body = "\n".join(body_lines)

        fm = {
            "title": title,
            "kind": "topic",
            "description": summary.split(". ")[0][:200] + ".",
            "source": "topic-clustering",
            "confidence": 0.70,
            "member_pages": list(cluster),
            "member_count": len(cluster),
        }
        pid_out = await write_synthesis_page(
            wiki_dir=wiki_dir, bm25=bm25, dense=dense,
            inputs=SynthesisPageInputs(title=title, body=body, frontmatter=fm,
                                       page_kind="topic", slug_prefix="topic"),
        )
        if pid_out:
            written.append(pid_out)

    log.info(
        "topic pages rebuilt",
        extra={"metadata": {
            "pages_scanned": len(embedded), "clusters": len(clusters),
            "removed_old": removed, "topics_written": len(written),
        }},
    )
    return {
        "pages_scanned": len(embedded),
        "clusters": len(clusters),
        "removed_old": removed,
        "topics_written": len(written),
        "pages": written,
    }
