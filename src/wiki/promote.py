"""Episodic → semantic auto-promotion.

Walks recent `wiki/episodic/<date>.md` entries and identifies clusters of
related queries that recur ≥ `min_repeats` times. For each qualifying cluster,
generates a synthesis page in `wiki/sources/auto-<slug>.md` so the recurring
topic crystallises from transient episodic into durable semantic memory.

Triggered by the daily scheduler (Phase D) and by the manual
`POST /admin/run/promote` endpoint.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..llm import OllamaClient
from .episodic import episodic_dir
from .synth_page import SynthesisPageInputs, write_synthesis_page

log = logging.getLogger(__name__)


_ENTRY_RE = re.compile(
    r"^## \[(?P<time>\d\d:\d\d:\d\dZ)\] (?P<kind>\w[\w\-]*) — (?P<title>.+?)$",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{4,}")
_PROMOTE_SYSTEM = (
    "You are consolidating a set of recurring queries from a research wiki's episodic log "
    "into ONE durable Markdown page. Produce: (1) a clear title, (2) a 1-sentence summary, "
    "(3) a 'Themes' section enumerating the common questions, (4) a 'Common answers' section "
    "with key points, (5) a 'See also' bullet list of cited wiki pages. Cite the underlying "
    "source pages using `[[stem|Title]]` wiki-links. "
    "Reply ONLY JSON: "
    '{"title":"…","summary":"…","themes":["…"],"key_points":["…"],"sources":["page-id"]}'
)


@dataclass
class _EpisodeRow:
    date: str
    time: str
    kind: str
    title: str
    body: str


def _read_recent_episodes(wiki_dir: Path, days: int) -> list[_EpisodeRow]:
    """Walk last `days` of episodic files and split into rows."""
    out: list[_EpisodeRow] = []
    today = date.today()
    d = episodic_dir(wiki_dir)
    for offset in range(days):
        target = today - timedelta(days=offset)
        f = d / f"{target.isoformat()}.md"
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        # Find headers + their character ranges → slice bodies between them.
        matches = list(_ENTRY_RE.finditer(text))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append(
                _EpisodeRow(
                    date=target.isoformat(),
                    time=m.group("time"),
                    kind=m.group("kind"),
                    title=m.group("title"),
                    body=text[start:end].strip(),
                )
            )
    return out


def _topic_signature(title: str, body: str) -> set[str]:
    """Cheap topic signature: deduplicated 4+ char tokens from title+body."""
    text = f"{title}\n{body}".lower()
    toks = {t for t in _TOKEN_RE.findall(text) if t not in _STOP}
    return toks


_STOP = {
    "this", "that", "with", "from", "have", "what", "when", "where", "which",
    "their", "would", "could", "should", "about", "there", "these", "those",
    "into", "than", "then", "such", "your", "yours", "them", "they", "more",
    "ingest", "query", "page", "title", "kind", "metadata", "confidence",
    "page_id", "correlation_id", "live", "true", "false",
}


def _cluster_episodes(
    rows: list[_EpisodeRow],
    *,
    overlap_threshold: float = 0.3,
    min_repeats: int,
) -> list[list[_EpisodeRow]]:
    """Greedy single-pass cluster: each row joins the first cluster whose
    signature has Jaccard overlap >= threshold; else starts a new cluster.
    Returns clusters of size >= min_repeats only."""
    sigs: list[set[str]] = []
    clusters: list[list[_EpisodeRow]] = []
    for row in rows:
        sig = _topic_signature(row.title, row.body)
        if not sig:
            continue
        joined = False
        for i, existing in enumerate(sigs):
            inter = len(sig & existing)
            union = len(sig | existing)
            if union and inter / union >= overlap_threshold:
                clusters[i].append(row)
                sigs[i] = existing | sig
                joined = True
                break
        if not joined:
            sigs.append(sig)
            clusters.append([row])
    return [c for c in clusters if len(c) >= min_repeats]


def _extract_correlation_ids(body: str) -> list[str]:
    return re.findall(r"`(COR-[A-Z0-9]{8,})`", body)


def _format_cluster_for_prompt(cluster: list[_EpisodeRow]) -> str:
    lines = []
    for r in cluster[:20]:
        lines.append(f"### [{r.date} {r.time}] {r.kind} — {r.title}")
        lines.append(r.body[:600])
        lines.append("")
    return "\n".join(lines)


def _build_promoted_body(parsed: dict, cluster: list[_EpisodeRow]) -> str:
    out = []
    out.append(f"# {parsed.get('title', 'Promoted topic')}")
    out.append("")
    if parsed.get("summary"):
        out.append(f"**TL;DR:** {parsed['summary']}")
        out.append("")
    out.append("## Themes")
    out.append("")
    for t in parsed.get("themes") or []:
        out.append(f"- {t}")
    out.append("")
    out.append("## Key points")
    out.append("")
    for k in parsed.get("key_points") or []:
        out.append(f"- {k}")
    out.append("")
    out.append("## Source episodes")
    out.append("")
    for r in cluster:
        out.append(f"- [{r.date} {r.time}] **{r.kind}** — {r.title}")
    out.append("")
    sources = parsed.get("sources") or []
    if sources:
        out.append("## See also")
        out.append("")
        for s in sources:
            stem = Path(str(s)).stem
            out.append(f"- [[{stem}]]")
    return "\n".join(out)


def _extract_json(s: str) -> dict | None:
    """Extract first balanced JSON object from a possibly noisy LLM response."""
    import json
    s = re.sub(r"^```(?:json)?\n?", "", (s or "").strip())
    s = re.sub(r"\n?```$", "", s)
    # Walk characters tracking brace depth — handles nested objects properly,
    # avoids the greedy `\{.*\}` failure mode where two objects in the response
    # get concatenated into invalid JSON.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


async def promote_episodic_to_semantic(
    *,
    wiki_dir: Path,
    bm25,
    dense,
    client: OllamaClient,
    days: int = 14,
    min_repeats: int = 3,
    max_clusters: int = 5,
) -> dict:
    """Top-level entry: scan episodic, cluster, and promote qualifying topics.

    Returns: {scanned, clusters, promoted, pages: [page_id]}.
    """
    rows = _read_recent_episodes(Path(wiki_dir), days=days)
    if not rows:
        return {"scanned": 0, "clusters": 0, "promoted": 0, "pages": []}

    clusters = _cluster_episodes(rows, min_repeats=min_repeats)
    promoted: list[str] = []
    for cluster in clusters[:max_clusters]:
        prompt = (
            f"Cluster of {len(cluster)} recurring episodes:\n\n"
            f"{_format_cluster_for_prompt(cluster)}\n\n"
            "Consolidate."
        )
        try:
            raw = await client.qwen(prompt, system=_PROMOTE_SYSTEM, temperature=0.2)
            parsed = _extract_json(raw) or {}
        except Exception as e:
            log.debug("promote LLM call failed", extra={"metadata": {"error": str(e)[:160]}})
            continue
        title = str(parsed.get("title") or cluster[0].title)[:120]
        body = _build_promoted_body(parsed, cluster)
        cor_ids = sorted({cid for r in cluster for cid in _extract_correlation_ids(r.body)})
        fm = {
            "title": title,
            "kind": "promoted",
            "source": "episodic-promotion",
            "source_episodes": [
                {"date": r.date, "time": r.time, "kind": r.kind, "title": r.title}
                for r in cluster
            ],
            "correlation_ids": cor_ids[:50],
            "confidence": 0.65,
            "created": datetime.now(timezone.utc).date().isoformat(),
        }
        pid = await write_synthesis_page(
            wiki_dir=Path(wiki_dir),
            bm25=bm25,
            dense=dense,
            inputs=SynthesisPageInputs(
                title=title, body=body, frontmatter=fm,
                page_kind="promoted", slug_prefix="auto",
            ),
        )
        if pid:
            promoted.append(pid)

    log.info(
        "episodic→semantic promotion",
        extra={"metadata": {
            "scanned": len(rows),
            "clusters": len(clusters),
            "promoted": len(promoted),
        }},
    )
    return {
        "scanned": len(rows),
        "clusters": len(clusters),
        "promoted": len(promoted),
        "pages": promoted,
    }
