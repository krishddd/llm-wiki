"""Query pipeline.

Advanced techniques wired in (all optional, controllable via flags):

- **HyDE** (Hypothetical Document Embeddings, Gao et al. 2022): generate a short
  hypothetical answer with the LLM, use its embedding (not the raw query) for the
  dense side of hybrid search. Improves recall on queries phrased differently from
  the source vocabulary.
- **Query decomposition**: for compound questions ("X and Y", "compare A with B"),
  split into sub-queries, retrieve for each, union → dedupe → rerank.
- **Self-grounding verification**: after synth, verify every `[citation]` token in
  the answer points to a retrieved page. Downgrade confidence on ungrounded claims.
- **Save-back**: high-confidence syntheses (≥ SAVE_BACK_CONF, ≥ 2 citations) persist
  as new `wiki/sources/synthesis-*.md` pages so valuable connections aren't lost.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .llm import OllamaClient, get_client
from .search.hybrid import hybrid_search
from .search.intent import profile_for
from .search.multi_query import paraphrase, rrf_fuse_pages
from .search.multimodal import extract_excerpts, truncate_block
from .search.relevance_eval import decide as crag_decide
from .search.relevance_eval import evaluate_batch as crag_evaluate
from .synth.blocks import AnswerBlock, number_citations, parse_blocks
from .synth.claims import Claim, aggregate_confidence, parse_claims, strip_confidence_markers
from .synth.followups import suggest_followups
from .synth.reflect import critique_answer, should_refine

log = logging.getLogger(__name__)

# ───── Prompts ─────

SYNTH_SYSTEM = (
    "You are a careful research assistant producing a NotebookLM-style structured answer. "
    "Use ONLY the provided wiki pages. If they don't contain the answer, say so honestly.\n\n"
    "FORMATTING RULES (must follow):\n"
    "- Use Markdown: '##' headings to break the answer into 2-4 sections when the question warrants it.\n"
    "- Use bullet lists for enumerations (3+ items).\n"
    "- When a page provides a [TABLE from <page>] block relevant to the question, REPRODUCE that table in your answer "
    "(GFM pipe-table format) — do not just describe it.\n"
    "- Render mathematical formulas as inline `$E=mc^2$` or display `$$ ... $$` LaTeX.\n"
    "- Use fenced code blocks with the correct language tag for code (```python, ```yaml, ```sql).\n"
    "- Use blockquote callouts `> [!note]` / `> [!warning]` / `> [!tip]` for important asides.\n"
    "- Cite every factual claim with [Page Title] in brackets — use the EXACT title as given. "
    "Multiple citations: [Page A][Page B]. Aim for at LEAST one citation per factual sentence; "
    "uncited claims will be flagged as ungrounded and rejected.\n"
    "- After each citation, append a confidence score for that specific claim in the form "
    "`[Page Title]^0.NN` (e.g. `[Active Inference For Ai Safety]^0.92`). Use 0.90+ for "
    "claims directly supported by clear text in the page; 0.7-0.85 for inferred claims; "
    "0.5-0.7 for tentative interpretations. NEVER OMIT the score.\n"
    "- For comparison/synthesis questions, structure the answer as: a short overview paragraph, "
    "then a Markdown comparison table OR per-source subsections, then a 'Key takeaways' bullet list.\n"
    "- For 'list all' / 'enumerate' questions, return a comprehensive bulleted list grouped by source page.\n"
    "- For factual questions, lead with a one-sentence direct answer; supporting context after.\n\n"
    "Reply ONLY with valid JSON matching this schema:\n"
    '{"answer":"<markdown with rich formatting and [Page Title] citations>",'
    '"summary":"<one-sentence TL;DR>",'
    '"key_points":["…","…"],'
    '"entities":["…"],'
    '"confidence":0.XX}\n'
    "Do not wrap the JSON in code fences. Do not add commentary outside the JSON."
)

HYDE_SYSTEM = (
    "Given a question, write 2-3 sentences of a hypothetical answer as if pretending to know. "
    "Use concrete nouns and domain-specific terms. The text will be embedded for search; quality of prose does not matter. "
    "Reply with plain text only, no preamble."
)

DECOMPOSE_SYSTEM = (
    "Decompose the user's question into up to 3 atomic sub-queries, each answerable independently. "
    "If the question is already atomic, reply with a single-element list. "
    'Reply ONLY JSON: {"sub_queries":["…","…"]}'
)

# ───── Config knobs ─────

SAVE_BACK_CONF = 0.80  # synthesis page persisted if confidence ≥ this AND ≥2 citations
DECOMPOSE_TRIGGERS = re.compile(
    r"\b(and|compare|vs\.?|versus|difference|both|between|how do .* relate|list .* of)\b", re.IGNORECASE
)


# ───── Data classes ─────

@dataclass
class CitationExcerpt:
    """Multimodal block surfaced inside a Citation (table/image/code)."""
    kind: str           # "table" | "image" | "code"
    content: str        # ready-to-render Markdown
    meta: dict = field(default_factory=dict)


@dataclass
class Citation:
    page: str
    title: str
    snippet: str
    has_tables: bool = False
    has_images: bool = False
    has_code: bool = False
    excerpts: list[CitationExcerpt] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str
    answer_raw: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    blocks: list[AnswerBlock] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    correlation_id: str = ""
    retrieved_pages: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    grounded: bool = True
    saved_page: str | None = None
    retrieval_quality: str = "correct"
    intent: str = "synthesis"
    quality_score: float = 1.0
    quality_issues: list[str] = field(default_factory=list)
    per_claim_confidences: list[dict] = field(default_factory=list)


def _extract_json(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?\n?", "", s.strip())
    s = re.sub(r"\n?```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_context(retrieved, query: str, *, full_page_mode: bool = False) -> tuple[str, list[Citation]]:
    """Assemble the LLM context from retrieved pages.

    For each page:
      - text snippet up to 1500 chars (or 6000 chars in full_page_mode for
        synthesis-class questions that need long-context evidence)
      - if the page has tables / images / code blocks, attach the top-scoring 1-2
        of each kind to the citation AND include the table markdown in the LLM's
        context so it can cite specific rows.

    Returns (context_string, citations).
    """
    blocks = []
    cits: list[Citation] = []
    snippet_budget = 6000 if full_page_mode else 1500
    citation_snippet_budget = 500 if full_page_mode else 300
    for r in retrieved:
        meta = r.meta or {}
        title = meta.get("title") or r.page_id
        full_body = r.text or ""
        snippet = full_body[:snippet_budget]

        # Pull multimodal excerpts ranked against the query.
        mm_raw = extract_excerpts(full_body, query=query, max_per_kind=2)
        excerpts = [
            CitationExcerpt(
                kind=e.kind,
                content=truncate_block(e.content, e.kind, max_chars=1200),
                meta=e.meta,
            )
            for e in mm_raw
        ]
        # also widen the per-citation snippet for synthesis questions
        _ = citation_snippet_budget
        has_tables = any(e.kind == "table" for e in excerpts) or bool(meta.get("has_tables"))
        has_images = any(e.kind == "image" for e in excerpts) or bool(meta.get("has_images"))
        has_code = any(e.kind == "code" for e in excerpts)

        # Build the LLM-facing block. Append top tables verbatim so the synth model
        # can cite specific values rather than paraphrasing.
        ctx_parts = [f"### [{title}] (id={r.page_id})", snippet]
        for e in excerpts:
            if e.kind == "table":
                ctx_parts.append(f"\n[TABLE from {title}]\n{e.content}")
            elif e.kind == "image":
                ctx_parts.append(f"\n[IMAGE from {title}] {e.content}")
            elif e.kind == "code":
                ctx_parts.append(f"\n[CODE from {title}]\n{e.content}")
        blocks.append("\n".join(ctx_parts))

        cits.append(
            Citation(
                page=r.page_id,
                title=str(title),
                snippet=snippet[:citation_snippet_budget],
                has_tables=has_tables,
                has_images=has_images,
                has_code=has_code,
                excerpts=excerpts,
            )
        )
    return "\n\n".join(blocks), cits


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s.strip().lower()).strip("-")
    return s[:80] or "synthesis"


def _check_grounded(answer: str, citations: list[Citation]) -> tuple[bool, int]:
    """Every [token] in answer should match a cited page title. Returns (grounded_flag, ungrounded_count)."""
    if not citations:
        return False, 0
    cited_titles = {c.title.strip().lower() for c in citations}
    # Match [bracketed tokens] that look like titles, skipping markdown links which have [text](url)
    tokens = re.findall(r"\[([^\]]+)\](?!\()", answer)
    if not tokens:
        return True, 0  # no citations asserted → trivially not contradicted
    ungrounded = 0
    for t in tokens:
        t_low = t.strip().lower()
        # Any cited title that overlaps the token substring-wise counts as grounded.
        if not any(t_low in ct or ct in t_low for ct in cited_titles):
            ungrounded += 1
    return ungrounded == 0, ungrounded


# ───── Engine ─────

class QueryEngine:
    def __init__(
        self,
        *,
        bm25,
        dense,
        page_store,
        graph=None,
        settings: Settings | None = None,
        client: OllamaClient | None = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.page_store = page_store
        self.graph = graph
        self.s = settings or get_settings()
        self.c = client or get_client()

    # ── Advanced pre-retrieval helpers ──

    async def _hyde(self, question: str) -> str | None:
        """Generate a hypothetical answer to use as the dense-embedding seed."""
        try:
            text = await self.c.qwen(question, system=HYDE_SYSTEM, temperature=0.5)
            return text.strip() if text else None
        except Exception as e:
            log.warning("HyDE generation failed, falling back to raw query", extra={"metadata": {"error": str(e)[:200]}})
            return None

    async def _decompose(self, question: str) -> list[str]:
        """Split compound questions. Cheap Qwen call, only fires if heuristic matches."""
        if not DECOMPOSE_TRIGGERS.search(question) or len(question) < 25:
            return [question]
        try:
            raw = await self.c.qwen(question, system=DECOMPOSE_SYSTEM, temperature=0.1)
            data = _extract_json(raw) or {}
            subs = [str(q).strip() for q in (data.get("sub_queries") or []) if str(q).strip()]
            return subs[:3] if subs else [question]
        except Exception as e:
            log.warning("decompose failed", extra={"metadata": {"error": str(e)[:200]}})
            return [question]

    async def _retrieve_one(self, query_text: str, top_k: int, graph_expand: bool, hyde_text: str | None):
        return await hybrid_search(
            query_text,
            bm25=self.bm25,
            dense=self.dense,
            page_store=self.page_store,
            graph=self.graph,
            top_k_rerank=top_k,
            graph_expand=graph_expand,
            hyde_text=hyde_text,
        )

    # ── Save-back ──

    async def _save_synthesis_page(self, question: str, result_data: dict, citations: list[Citation]) -> str | None:
        """Save-back wrapper. Delegates to the standalone util in `wiki.synth_page`
        so the same page-shape is produced by save-back / promotion / crystallization."""
        try:
            body_lines = [f"# {question.strip()}", ""]
            if result_data.get("summary"):
                body_lines.append(f"**TL;DR:** {result_data['summary']}")
                body_lines.append("")
            body_lines.append("## Answer")
            body_lines.append("")
            body_lines.append(str(result_data.get("answer", "")).strip())
            body_lines.append("")
            if result_data.get("key_points"):
                body_lines.append("## Key Points")
                body_lines.append("")
                for kp in result_data["key_points"]:
                    body_lines.append(f"- {kp}")
                body_lines.append("")
            body_lines.append("## Sources")
            body_lines.append("")
            for c in citations:
                stem = Path(c.page).stem
                body_lines.append(f"- [[{stem}|{c.title}]]")
            body = "\n".join(body_lines)

            frontmatter = {
                "title": question.strip()[:120],
                "kind": "synthesis",
                "source": "query-save-back",
                "confidence": round(float(result_data.get("confidence", 0.0)), 2),
                "entity_refs": list(result_data.get("entities") or [])[:30],
                "cited_pages": [c.page for c in citations],
                "created": date.today().isoformat(),
            }

            from .wiki.synth_page import SynthesisPageInputs, write_synthesis_page
            pid = await write_synthesis_page(
                wiki_dir=self.s.wiki_dir,
                bm25=self.bm25,
                dense=self.dense,
                inputs=SynthesisPageInputs(
                    title=question.strip()[:120],
                    body=body,
                    frontmatter=frontmatter,
                    page_kind="synthesis",
                ),
            )
            return pid
        except Exception as e:
            log.warning("save-back failed", extra={"metadata": {"error": str(e)[:200]}})
            return None

    # ── Main ──

    async def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        graph_expand: bool = True,
        use_hyde: bool = True,
        decompose: bool = True,
        save_back: bool = True,
        adaptive: bool | None = None,
    ) -> QueryResult:
        # 0) Adaptive retrieval profile — pick top_k / full-page mode / graph
        # expand by question intent. Heuristic first, gemma fallback.
        adaptive_on = self.s.query_adaptive_retrieval if adaptive is None else adaptive
        if adaptive_on:
            try:
                prof = await profile_for(self.c, question, default_top_k=top_k)
                top_k = prof.top_k
                graph_expand = graph_expand and prof.graph_expand
                full_page_mode = prof.full_page_mode
                intent_label = prof.intent
                log.info(
                    "intent profile",
                    extra={"metadata": {
                        "intent": prof.intent, "top_k": top_k,
                        "full_page": full_page_mode, "rationale": prof.rationale,
                    }},
                )
            except Exception as e:
                log.debug("intent profile failed", extra={"metadata": {"error": str(e)[:120]}})
                full_page_mode = False
                intent_label = "synthesis"
        else:
            full_page_mode = False
            intent_label = "synthesis"

        crag_overall = "correct"
        crag_ceiling = 1.0

        # Active Procedural Execution: Check if this question matches a crystallized procedure
        matched_procedure = None
        if self.procedures:
            try:
                from .wiki.procedures import _pattern_hash
                ph = _pattern_hash(question)
                cur = self.procedures._conn.cursor()
                cur.execute(
                    "SELECT query_template, canonical_pages, hit_count, promoted_at "
                    "FROM procedures WHERE pattern_hash = ?",
                    (ph,),
                )
                row = cur.fetchone()
                if row and row[1]:
                    import json
                    canonical_pages = json.loads(row[1])
                    if canonical_pages:
                        matched_procedure = {
                            "query_template": row[0],
                            "canonical_pages": canonical_pages,
                            "hit_count": row[2],
                            "promoted": bool(row[3])
                        }
            except Exception as e:
                log.debug("Procedural recall check failed", extra={"metadata": {"error": str(e)[:120]}})

        retrieved = []
        if matched_procedure:
            log.info(
                "procedural recall triggered",
                extra={"metadata": {
                    "query": question,
                    "template": matched_procedure["query_template"],
                    "hits": matched_procedure["hit_count"],
                    "pages": matched_procedure["canonical_pages"]
                }},
            )
            # Retrieve specified anchor pages directly from the page_store
            for pid in matched_procedure["canonical_pages"][:top_k]:
                text = await self.page_store.get_text(pid)
                if text:
                    meta = await self.page_store.get_meta(pid)
                    from .search.hybrid import RetrievedPage
                    retrieved.append(RetrievedPage(page_id=pid, text=text[:6000], score=1.0, meta=meta or {}))
        else:
            # 1) Query decomposition (optional, heuristic-gated).
            sub_queries = await self._decompose(question) if decompose else [question]

            # 2) HyDE: hallucinate an answer; its embedding drives dense retrieval.
            hyde_text = await self._hyde(question) if use_hyde else None

            # 3) Per sub-query: optionally paraphrase (RAG-Fusion), retrieve each
            #    paraphrase, RRF-fuse the page-id lists, then re-fetch the merged
            #    set's data from the most recent batch. Track the best rerank score
            #    per page across all paraphrases.
            merged: dict[str, Any] = {}
            for sq in sub_queries:
                queries_for_sq: list[str] = [sq]
                if self.s.query_multi_query:
                    try:
                        queries_for_sq = await paraphrase(self.c, sq, max_paraphrases=2)
                    except Exception as e:
                        log.debug("paraphrase failed", extra={"metadata": {"error": str(e)[:120]}})

                ranked_lists: list[list[str]] = []
                page_objs: dict[str, Any] = {}
                for q in queries_for_sq:
                    batch = await self._retrieve_one(q, top_k=top_k, graph_expand=graph_expand, hyde_text=hyde_text)
                    ranked_lists.append([r.page_id for r in batch])
                    for r in batch:
                        prev = page_objs.get(r.page_id)
                        if prev is None or r.score > prev.score:
                            page_objs[r.page_id] = r

                # RRF-fuse the multiple ranked lists, then re-attach scores.
                fused_ids = rrf_fuse_pages(ranked_lists)[:top_k] if len(ranked_lists) > 1 else ranked_lists[0][:top_k]
                for pid in fused_ids:
                    obj = page_objs.get(pid)
                    if obj is None:
                        continue
                    prev = merged.get(pid)
                    if prev is None or obj.score > prev.score:
                        merged[pid] = obj
            retrieved = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:top_k]

            # 3b) CRAG-style relevance evaluation. Drop pages flagged "incorrect";
            # cap the final confidence if all surviving pages are merely "ambiguous".
            if retrieved and self.s.query_relevance_eval:
                try:
                    verdicts = await crag_evaluate(
                        self.c, question, [(r.page_id, r.text or "") for r in retrieved],
                    )
                    decision = crag_decide(verdicts)
                    crag_overall = decision.overall
                    crag_ceiling = decision.confidence_ceiling
                    keep = set(decision.keep_page_ids)
                    if keep and len(keep) < len(retrieved):
                        log.info(
                            "CRAG dropped off-topic pages",
                            extra={"metadata": {
                                "before": len(retrieved), "after": len(keep), "overall": crag_overall,
                            }},
                        )
                        retrieved = [r for r in retrieved if r.page_id in keep]
                    if not retrieved:
                        log.info("CRAG dropped all retrieved pages — answering as 'no relevant info'")
                except Exception as e:
                    log.debug("CRAG eval failed", extra={"metadata": {"error": str(e)[:120]}})

        if not retrieved:
            empty_msg = "I couldn't find any relevant wiki pages for that question."
            return QueryResult(
                answer=empty_msg,
                answer_raw=empty_msg,
                summary="No relevant pages in the wiki.",
                key_points=[],
                citations=[],
                blocks=[AnswerBlock(kind="text", content=empty_msg)],
                follow_up_questions=[],
                entities=[],
                confidence=0.0,
                retrieved_pages=[],
                sub_queries=sub_queries if not matched_procedure else [question],
                grounded=True,
            )

        # 4) Synthesis with strict JSON schema + retry.
        ctx, cits = _build_context(retrieved, query=question, full_page_mode=full_page_mode)

        # GraphRAG: Compile structured "Canonical Truth" table of active bi-temporal facts
        active_facts = []
        seen_facts = set()
        if self.graph:
            entities_to_query = []
            for r in retrieved:
                meta = r.meta or {}
                for ent in meta.get("entity_refs") or []:
                    if ent not in entities_to_query:
                        entities_to_query.append(ent)

            for ent in entities_to_query[:15]:
                try:
                    facts = await self.graph.active_facts_for(ent)
                    for f in facts:
                        key = (ent.lower(), f["predicate"].lower(), f["object"].lower())
                        if key not in seen_facts:
                            seen_facts.add(key)
                            active_facts.append(
                                f"- **{ent}** {f['predicate']} *{f['object']}* (Source: {f['source_page']})"
                            )
                except Exception as e:
                    log.debug("GraphRAG facts query failed", extra={"metadata": {"entity": ent, "error": str(e)[:120]}})

        fact_context = ""
        if active_facts:
            fact_context = "\n\nCANONICAL GRAPH FACTS (VERIFIED ACTIVE TRUTH — CITE THESE WHEN POSSIBLE):\n" + "\n".join(active_facts)

        prompt = f"QUESTION:\n{question}\n\nWIKI PAGES:\n{ctx}{fact_context}"
        raw = await self.c.qwen(prompt, system=SYNTH_SYSTEM, temperature=0.2)
        data = _extract_json(raw)
        if data is None:
            fixup = f"Your previous reply was not valid JSON. Reply ONLY with valid JSON now.\n\n{raw}"
            raw = await self.c.qwen(fixup, system=SYNTH_SYSTEM, temperature=0.1)
            data = _extract_json(raw) or {}

        answer_text = str(data.get("answer", raw))[:8000]
        confidence = float(data.get("confidence", 0.5))

        # 5) Self-grounding check. Penalise confidence if claims aren't backed by citations.
        grounded, ungrounded_count = _check_grounded(answer_text, cits)
        if not grounded:
            penalty = min(0.4, 0.1 * ungrounded_count)
            confidence = max(0.0, confidence - penalty)
            log.warning(
                "ungrounded claims detected",
                extra={"metadata": {"ungrounded_count": ungrounded_count, "penalty": penalty}},
            )

        # 5b) CRAG ceiling: if retrieval was weak/ambiguous, never let final
        # confidence exceed `crag_ceiling`. Prevents the synthesizer from
        # hand-waving a high score on top of mediocre evidence.
        if confidence > crag_ceiling:
            log.info(
                "CRAG ceiling applied",
                extra={"metadata": {"from": confidence, "to": crag_ceiling, "overall": crag_overall}},
            )
            confidence = crag_ceiling

        # 5c) Reflection / critique pass. If the draft is incomplete and we're
        # allowed to refine, do ONE re-synthesis pass with the gap guidance.
        critique_score = 1.0
        critique_issues: list[str] = []
        if self.s.query_reflect and answer_text:
            try:
                crit = await critique_answer(
                    self.c,
                    question=question,
                    answer=answer_text,
                    source_titles=[c.title for c in cits],
                )
                critique_score = crit.quality_score
                critique_issues = crit.issues
                if self.s.query_reflect_refine and should_refine(crit):
                    log.info(
                        "reflection: triggering refinement",
                        extra={"metadata": {
                            "quality": crit.quality_score,
                            "missing": crit.missing_aspects[:3],
                        }},
                    )
                    refine_prompt = (
                        f"QUESTION:\n{question}\n\n"
                        f"WIKI PAGES:\n{ctx}\n\n"
                        f"PREVIOUS DRAFT:\n{answer_text[:3000]}\n\n"
                        f"REFINE GUIDANCE: {crit.suggested_refinement}\n"
                        f"Address the missing aspects: {', '.join(crit.missing_aspects[:3])}\n"
                        "Produce an improved answer following the formatting rules. JSON only."
                    )
                    try:
                        raw2 = await self.c.qwen(refine_prompt, system=SYNTH_SYSTEM, temperature=0.2)
                        data2 = _extract_json(raw2)
                        if data2 and data2.get("answer"):
                            answer_text = str(data2.get("answer", answer_text))[:8000]
                            # Re-check grounding on the refined answer.
                            grounded, ungrounded_count = _check_grounded(answer_text, cits)
                            if "summary" in data2:
                                data["summary"] = data2["summary"]
                            if "key_points" in data2:
                                data["key_points"] = data2["key_points"]
                    except Exception as e:
                        log.debug("refinement synth failed", extra={"metadata": {"error": str(e)[:120]}})
            except Exception as e:
                log.debug("reflection failed", extra={"metadata": {"error": str(e)[:120]}})

        # 6) Save-back: persist valuable syntheses as wiki pages.
        saved = None
        if save_back and confidence >= SAVE_BACK_CONF and len(cits) >= 2:
            saved = await self._save_synthesis_page(
                question,
                {
                    "answer": answer_text,
                    "summary": str(data.get("summary", "")),
                    "key_points": data.get("key_points") or [],
                    "entities": data.get("entities") or [],
                    "confidence": confidence,
                },
                cits,
            )

        # 7) NotebookLM-style post-processing.
        # 7a) Parse per-claim confidence markers `[Page]^0.NN` BEFORE numbering.
        per_claim: list[Claim] = []
        if self.s.query_per_claim_confidence:
            per_claim = parse_claims(answer_text)
            if per_claim:
                # Calibrate the overall confidence with per-claim aggregate.
                claim_overall = aggregate_confidence(per_claim, ceiling=crag_ceiling)
                # Final confidence = mean of model's own + claim aggregate (smoothing).
                confidence = round(0.5 * confidence + 0.5 * claim_overall, 3)
                # Strip the `^0.NN` markers from the user-facing answer.
                answer_text = strip_confidence_markers(answer_text)

        #    b) Number citations [Title] → [1], aligned to citation order in `cits`.
        cite_titles = [c.title for c in cits]
        numbered_answer, _appearance = number_citations(answer_text, cite_titles)
        #    b) Parse the numbered answer into typed blocks.
        try:
            blocks = parse_blocks(numbered_answer)
        except Exception as e:
            log.warning("blocks parse failed", extra={"metadata": {"error": str(e)[:160]}})
            blocks = [AnswerBlock(kind="text", content=numbered_answer)]
        #    c) Suggest follow-ups (best-effort, never blocks the response).
        try:
            follow_ups = await suggest_followups(
                self.c,
                question=question,
                answer_summary=str(data.get("summary", ""))[:500] or numbered_answer[:500],
                page_titles=cite_titles,
            )
        except Exception:
            follow_ups = []

        # Phase C4: record query pattern in the procedural memory store. Best-effort.
        procedures = getattr(self, "procedures", None)
        if procedures is not None and grounded and confidence >= 0.5:
            try:
                from .wiki.procedures import record_query_pattern
                await record_query_pattern(
                    procedures, question, intent_label,
                    [r.page_id for r in retrieved],
                )
            except Exception as e:
                log.debug("procedure record failed",
                          extra={"metadata": {"error": str(e)[:120]}})

        return QueryResult(
            answer=numbered_answer,
            answer_raw=answer_text,
            summary=str(data.get("summary", ""))[:500],
            key_points=[str(k) for k in (data.get("key_points") or [])][:20],
            citations=cits,
            blocks=blocks,
            follow_up_questions=follow_ups,
            entities=[str(e) for e in (data.get("entities") or [])][:30],
            confidence=round(confidence, 3),
            retrieved_pages=[r.page_id for r in retrieved],
            sub_queries=sub_queries,
            grounded=grounded,
            saved_page=saved,
            retrieval_quality=crag_overall,
            intent=intent_label,
            quality_score=round(critique_score, 3),
            quality_issues=critique_issues,
            per_claim_confidences=[
                {"citation": c.citation_token, "confidence": round(c.confidence, 3)}
                for c in per_claim
            ],
        )
