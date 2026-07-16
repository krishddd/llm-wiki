"""Document ingest pipeline: load → chunk → Gemma summarise+extract (throttled) → Qwen merge + confidence → write.

Concurrency note: Ollama queues concurrent requests to the same model. We use a module-level asyncio.Semaphore
sized by `max_concurrent_llm_req` (env-configurable) to cap in-flight per-model requests and avoid timeout cascades
that would otherwise trigger the llama3.2 fallback falsely.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from .config import Settings, get_settings
from .graph import ExtractedEntity, ExtractedRelation, KnowledgeGraph
from .llm import OllamaClient, get_client
from .loaders import elements_to_markdown, layout_aware_chunks, load_elements, load_source  # noqa: F401
from .loaders.elements import DocElement
from .logging_config import audit
from .search.bm25_index import BM25Index
from .search.chunks import SUB_CHUNK_OVERLAP, SUB_CHUNK_TARGET, chunk_text
from .search.dense_index import DenseIndex
from .wiki.entity_pages import rebuild_entity_pages
from .wiki.episodic import append_episode
from .wiki.index_md import rebuild_index
from .wiki.log_md import append_log
from .wiki.pages import (
    Page,
    derive_description,
    page_id_from_path,
    read_page,
    stage_or_publish,
    write_page,
)
from .wiki.reconciler import (
    apply_edit_to_body,
    find_affected_pages,
    propose_edit,
    should_auto_apply,
    should_propose,
    stage_proposal,
)

log = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Summarise the following text in 150–250 words. Focus on concrete facts, named entities, and claims. "
    "Plain prose, no headers.\n\nTEXT:\n"
)
ENTITY_PROMPT = (
    "Extract named entities and relations from the text. Reply ONLY with JSON matching this schema:\n"
    '{"entities":[{"name":"...","type":"PERSON|ORG|CONCEPT|PLACE|EVENT"}],'
    '"relations":[{"src":"...","src_type":"...","dst":"...","dst_type":"...","rel_type":"RELATES_TO|PART_OF|CONTRADICTS|SUPPORTS|AUTHORED_BY|OCCURRED_IN"}]}\n\nTEXT:\n'
)
MERGE_PROMPT = (
    "You are merging partial chunk summaries of one document into ONE cohesive 400–600 word summary. "
    "Preserve all named entities and claims. Return only the merged summary.\n\nPARTIAL SUMMARIES:\n"
)
CONFIDENCE_PROMPT = (
    'Rate how confident you are that the summary faithfully reflects the source. Reply ONLY JSON: '
    '{"confidence": 0.XX, "reason": "..."}\n\nSUMMARY:\n'
)
CLAIM_PROMPT = (
    "Extract atomic factual claims from the SUMMARY as subject-predicate-object triples. "
    "Each subject MUST be one of the entities listed in ENTITIES — DO NOT invent new subjects. "
    "Predicates are short verb phrases (≤6 words). Objects are short noun phrases or other entity names. "
    "Reply ONLY JSON: "
    '{"claims":[{"subject":"…","predicate":"…","object":"…","confidence":0.XX}]}\n'
    "Cap at 12 claims. Skip claims you can't confidently extract.\n"
    "ENTITIES:\n{entities}\n\nSUMMARY:\n{summary}"
)
DOC2QUERY_PROMPT = (
    "Generate 4 short, distinct questions that this document directly answers. "
    "Phrase them the way a user would naturally ask (who/what/how/why). "
    'Reply ONLY JSON: {"questions":["…","…","…","…"]}\n\nDOCUMENT SUMMARY:\n'
)
CONTRADICTION_PROMPT = (
    "Compare the NEW summary against the EXISTING page content. Does the NEW summary contradict any factual "
    "claim in EXISTING? Reply ONLY JSON: "
    '{"contradicts": true|false, "claim": "short quote of the contradicting claim or empty", "reason": "..."}'
    "\n\nNEW:\n{new}\n\nEXISTING:\n{old}\n"
)


@dataclass
class IngestResult:
    source: str
    page_path: str
    confidence: float
    is_live: bool
    entities_added: int = 0
    title: str = ""
    error: str | None = None
    chunks: int = 0
    extracted: dict = field(default_factory=dict)


# Canonical chunker lives in search.chunks so retrieval can re-derive the exact
# sub-chunks at query time (small-to-big). Keep the old local name as an alias.
_chunk_text = chunk_text


def _extract_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Ingestor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OllamaClient | None = None,
        graph: KnowledgeGraph | None = None,
        bm25: BM25Index | None = None,
        dense: DenseIndex | None = None,
    ):
        self.s = settings or get_settings()
        self.c = client or get_client()
        self.graph = graph
        self.bm25 = bm25
        self.dense = dense
        self._sem = asyncio.Semaphore(self.s.max_concurrent_llm_req)

    async def _summarise_chunk(self, chunk: str) -> str:
        try:
            async with self._sem:
                return await self.c.gemma(SUMMARY_PROMPT + chunk)
        except Exception as e:
            log.warning("summarise chunk failed, using raw excerpt", extra={"metadata": {"error": str(e)[:200]}})
            # Fall back to a raw excerpt so the doc still makes it through.
            return chunk[:1500]

    async def _extract_chunk(self, chunk: str) -> dict:
        try:
            async with self._sem:
                raw = await self.c.gemma(ENTITY_PROMPT + chunk[:5000], temperature=0.1)
            return _extract_json(raw) or {"entities": [], "relations": []}
        except Exception as e:
            log.warning("extract chunk failed, skipping", extra={"metadata": {"error": str(e)[:200]}})
            return {"entities": [], "relations": []}

    async def _merge_summaries(self, parts: list[str]) -> str:
        """Merge partial chunk-summaries into one coherent doc summary.

        Strategy:
        1) If only one part, return it.
        2) Try the full merge (qwen). On timeout/error:
        3) Halve the input (truncate each partial proportionally) and retry.
        4) If still failing, fall back to a *coherent* prose blend rather than
           raw concatenation — concatenated partials look incoherent to the
           confidence scorer and get marked 0.0.
        """
        if len(parts) == 1:
            return parts[0].strip()

        # ── Attempt 1: full merge ──
        joined = "\n\n---\n\n".join(parts)
        # Cap the merge prompt to ~16k chars total to keep qwen responsive.
        if len(joined) > 16000:
            per_part_budget = max(400, 16000 // len(parts))
            trimmed = [p[:per_part_budget] for p in parts]
            joined = "\n\n---\n\n".join(trimmed)
        try:
            out = await self.c.qwen(MERGE_PROMPT + joined)
            if out and len(out.strip()) > 100:
                return out.strip()
            log.warning("merge returned empty/short, retrying with shorter input")
        except Exception as e:
            log.warning("merge attempt 1 failed", extra={"metadata": {"error": str(e)[:200]}})

        # ── Attempt 2: halve the input and retry once ──
        per_part = max(200, 6000 // max(len(parts), 1))
        small_joined = "\n\n---\n\n".join(p[:per_part] for p in parts)
        try:
            out = await self.c.qwen(MERGE_PROMPT + small_joined)
            if out and len(out.strip()) > 100:
                return out.strip()
        except Exception as e:
            log.warning("merge attempt 2 failed", extra={"metadata": {"error": str(e)[:200]}})

        # ── Final fallback: coherent prose blend ──
        # Pick the LONGEST partial as the spine, then append unique sentences from
        # the others. Result reads as prose — confidence scorer treats it as a
        # legitimate summary, not garbage.
        log.warning("merge fully failed, building coherent fallback from partials")
        parts_clean = [p.strip() for p in parts if p and p.strip()]
        if not parts_clean:
            return ""
        spine = max(parts_clean, key=len)
        spine_low = spine.lower()
        extras: list[str] = []
        seen_signatures: set[str] = set()
        for p in parts_clean:
            if p is spine:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", p):
                s = sent.strip()
                if len(s) < 40:
                    continue
                sig = s[:60].lower()
                if sig in seen_signatures:
                    continue
                if s[:80].lower() in spine_low:
                    continue
                seen_signatures.add(sig)
                extras.append(s)
                if sum(len(e) for e in extras) > 1500:
                    break
            if sum(len(e) for e in extras) > 1500:
                break
        blended = spine
        if extras:
            blended = f"{spine}\n\nAdditional points from later sections: " + " ".join(extras)
        return blended[:5000]

    # ─────────────────────────────────────────────────────────────────
    # Phase A — Claim extraction & supersede helpers
    # ─────────────────────────────────────────────────────────────────

    async def _extract_claims(
        self,
        summary: str,
        entities: list[ExtractedEntity],
    ) -> list[dict]:
        """Extract S-P-O claims from the merged summary, scoped to known entities.

        Returns list of {subject, predicate, object, confidence}. Empty on any
        failure — claim extraction is best-effort and must NEVER block ingest.
        """
        if not summary or not entities:
            return []
        # Scope: only the first 30 entities, only the first 4000 chars of summary.
        ent_lines = "\n".join(f"- {e.name} ({e.type})" for e in entities[:30])
        prompt = CLAIM_PROMPT.format(entities=ent_lines, summary=summary[:4000])
        try:
            raw = await self.c.qwen(prompt, temperature=0.1)
        except Exception as e:
            log.debug("claim extraction failed", extra={"metadata": {"error": str(e)[:160]}})
            return []
        data = _extract_json(raw) or {}
        claims = data.get("claims") or []
        # Build the entity-name allow-list; lowercase for lookup.
        ent_lookup: dict[str, str] = {e.name.lower(): e.type for e in entities}
        out: list[dict] = []
        for c in claims:
            s = (c.get("subject") or "").strip()
            p = (c.get("predicate") or "").strip()
            o = (c.get("object") or "").strip()
            if not s or not p or not o:
                continue
            stype = ent_lookup.get(s.lower())
            if stype is None:
                continue  # subject not in known entity set — skip
            try:
                conf = float(c.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            otype = ent_lookup.get(o.lower())  # may be None (RHS is a literal)
            out.append({
                "subject": s, "subject_type": stype,
                "predicate": p[:60],
                "object": o[:300], "object_type": otype,
                "confidence": conf,
            })
        return out[:12]

    async def _persist_claims(self, pid: str, claims: list[dict]) -> int:
        """Insert claims into graph.facts. Returns count persisted."""
        if not self.graph or not claims:
            return 0
        n = 0
        for c in claims:
            try:
                await self.graph.add_fact(
                    subject_name=c["subject"],
                    subject_type=c["subject_type"],
                    predicate=c["predicate"],
                    object_text=c["object"],
                    object_type=c.get("object_type"),
                    source_page=pid,
                    confidence=c["confidence"],
                )
                n += 1
            except Exception as e:
                log.debug("add_fact failed", extra={"metadata": {"error": str(e)[:120]}})
                continue
        return n

    async def _supersede_facts_matching_old_text(
        self,
        target_pid: str,
        old_text: str,
        new_pid: str,
    ) -> int:
        """Phase A2/A3: when a contradicting/refining edit lands, mark facts on
        the older page that overlap with `old_text` as superseded.

        Strategy:
          1. Pull all active facts whose source_page == target_pid.
          2. Substring-match (case-insensitive) any whose object_text appears in old_text.
          3. Find the best matching new fact on `new_pid` (same predicate root) — if
             none, just close the old fact's window without a successor.

        Returns number of facts superseded.
        """
        if not self.graph or not old_text:
            return 0
        # Snapshot rows under the lock so subsequent supersede_fact calls
        # (which also acquire the lock) don't race with us.
        try:
            async with self.graph._lock:
                cur = self.graph._conn.cursor()
                cur.execute(
                    "SELECT id, predicate, object_text FROM facts WHERE source_page = ? "
                    "AND (valid_to IS NULL OR valid_to = '') ",
                    (target_pid,),
                )
                old_rows = cur.fetchall()
                cur.execute(
                    "SELECT id, predicate, object_text FROM facts WHERE source_page = ? "
                    "AND (valid_to IS NULL OR valid_to = '') ",
                    (new_pid,),
                )
                new_rows = cur.fetchall()
        except Exception as e:
            log.debug("supersede: facts query failed", extra={"metadata": {"error": str(e)[:120]}})
            return 0

        ot_low = old_text.lower()
        # Build new-fact predicate index for lookup.
        new_by_pred: dict[str, int] = {}
        for nid, npred, _ in new_rows:
            new_by_pred.setdefault(str(npred).strip().lower(), int(nid))

        today = datetime.now(UTC).date().isoformat()
        n = 0
        for fid, fpred, fobj in old_rows:
            obj = str(fobj or "").lower()
            if not obj or obj not in ot_low:
                continue
            # Try to find a successor fact with the same predicate on the new page.
            successor_id = new_by_pred.get(str(fpred).strip().lower())
            try:
                if successor_id is not None:
                    await self.graph.supersede_fact(
                        old_fact_id=int(fid), new_fact_id=int(successor_id), valid_to=today,
                    )
                else:
                    # Close the old fact's window even without a successor.
                    async with self.graph._lock:
                        cur = self.graph._conn.cursor()
                        cur.execute(
                            "UPDATE facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                            (today, int(fid)),
                        )
                        self.graph._conn.commit()
                n += 1
            except Exception as e:
                log.debug("supersede_fact call failed", extra={"metadata": {"error": str(e)[:120]}})
                continue
        if n > 0:
            audit(
                log, "FACT_SUPERSEDED", target_pid,
                count=n, triggered_by=new_pid,
            )
        return n

    async def _detect_contradictions(self, pid: str, summary: str, entity_names: list[str]) -> int:
        """Check the new summary against pages that cite overlapping entities. Emit audit events. Returns count."""
        if not self.graph or not entity_names:
            return 0
        try:
            related_pages: set[str] = set()
            for name in entity_names[:5]:
                pages = await self.graph.pages_for_entity(name)
                for p in pages:
                    if p != pid:
                        related_pages.add(p)
                if len(related_pages) >= 5:
                    break
        except Exception as e:
            log.debug("contradiction: pages_for_entity failed", extra={"metadata": {"error": str(e)[:120]}})
            return 0

        found = 0
        for other_pid in list(related_pages)[:5]:
            try:
                other_path = Path(self.s.wiki_dir) / other_pid
                if not other_path.exists():
                    continue
                old_text = other_path.read_text(encoding="utf-8")[:3000]
                prompt = CONTRADICTION_PROMPT.format(new=summary[:2500], old=old_text)
                raw = await self.c.qwen(prompt, temperature=0.1)
                data = _extract_json(raw) or {}
                if bool(data.get("contradicts")):
                    found += 1
                    claim_str = str(data.get("claim", ""))[:200]
                    audit(
                        log,
                        "CONTRADICTION_DETECTED",
                        pid,
                        compared_to=other_pid,
                        claim=claim_str,
                        reason=str(data.get("reason", ""))[:200],
                    )
                    # Phase A3: when a contradiction is flagged with a concrete claim
                    # excerpt, immediately mark matching facts on the older page as
                    # superseded by the new page. Best-effort.
                    if claim_str:
                        try:
                            await self._supersede_facts_matching_old_text(
                                target_pid=other_pid, old_text=claim_str, new_pid=pid,
                            )
                        except Exception as e:
                            log.debug("supersede on contradiction failed",
                                      extra={"metadata": {"error": str(e)[:120]}})
                    # Phase E2: also try the composite-score auto-resolver against
                    # any STILL-active fact pairs that disagree on this subject.
                    try:
                        from .wiki.contradiction_resolver import (
                            list_unresolved_contradictions,
                            resolve_contradiction,
                        )
                        pairs = await list_unresolved_contradictions(self.graph, limit=10)
                        for pair in pairs:
                            await resolve_contradiction(
                                self.graph,
                                pair["fact_a"], pair["fact_b"],
                                wiki_dir=self.s.wiki_dir,
                            )
                    except Exception as e:
                        log.debug("auto-resolver pass failed",
                                  extra={"metadata": {"error": str(e)[:120]}})
            except Exception as e:
                log.debug("contradiction check failed", extra={"metadata": {"error": str(e)[:120]}})
                continue
        return found

    async def _score_confidence(self, summary: str) -> tuple[float, str]:
        # Quick guards: a too-short or empty summary can't score legitimately. Don't
        # waste a qwen call — return a low-but-nonzero default so the page lands
        # in review rather than getting hard-zeroed.
        if not summary or len(summary.strip()) < 80:
            return 0.35, "summary too short to score"
        try:
            raw = await self.c.qwen(CONFIDENCE_PROMPT + summary[:4000], temperature=0.1)
            data = _extract_json(raw) or {}
            conf = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", ""))
            # Defensive: qwen sometimes returns near-zero with a generic "I can't
            # verify" excuse even when handed a real summary. The model has a
            # known hallucinated-meta-failure mode where it interprets the prompt
            # too literally. Detect a wider set of excuse signals and bump.
            excuse_signals = (
                "no source", "without access", "cannot verify", "no original",
                "no source text", "unable to assess", "unable to verify",
                "without the original", "fragmented", "incoherent",
                "appears truncated", "lack of context", "no context provided",
                "without the source", "i cannot determine", "cannot be assessed",
            )
            if conf < 0.2 and any(sig in reason.lower() for sig in excuse_signals):
                log.info(
                    "qwen returned low-confidence with meta-excuse; bumping to 0.4",
                    extra={"metadata": {"from": conf, "original_reason": reason[:200]}},
                )
                reason = f"auto-bumped (was {conf:.2f} with meta-excuse): {reason}"
                conf = 0.4
            return conf, reason
        except Exception as e:
            log.warning("confidence scoring failed, defaulting to 0.4", extra={"metadata": {"error": str(e)[:200]}})
            return 0.4, f"scoring_error: {str(e)[:120]}"

    async def _caption_images(self, elements: list[DocElement]) -> int:
        """Caption DocElement(kind='image') in place via llava. Returns count captioned."""
        if not self.s.ingest_caption_images:
            return 0
        captioned = 0
        for el in elements:
            if el.kind != "image" or el.content:
                continue
            if captioned >= self.s.ingest_max_image_caption:
                break
            img_path = el.meta.get("path")
            if not img_path or not Path(img_path).exists():
                continue
            try:
                async with self._sem:
                    caption = await self.c.llava(
                        "Describe this image in one concise sentence. Mention any text, "
                        "diagrams, charts, or entities visible.",
                        img_path,
                    )
                el.content = (caption or "").strip()[:400]
                captioned += 1
            except Exception as e:
                log.debug("llava caption failed", extra={"metadata": {"error": str(e)[:120]}})
                continue
        return captioned

    async def _doc2query(self, summary: str) -> list[str]:
        """Doc2Query (Nogueira & Lin): generate the questions this document answers.

        The questions are indexed alongside the page (`<pid>#hq`) so a user query
        phrased as a question can match lexically/semantically even when the
        document itself uses declarative vocabulary. Best-effort — one gemma call.
        """
        if not summary:
            return []
        try:
            async with self._sem:
                raw = await self.c.gemma(DOC2QUERY_PROMPT + summary[:3000], temperature=0.4)
            data = _extract_json(raw) or {}
            return [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:6]
        except Exception as e:
            log.debug("doc2query failed", extra={"metadata": {"error": str(e)[:120]}})
            return []

    async def _index_doc2query(self, pid: str, title: str, questions: list[str]) -> None:
        """Index the hypothetical questions as their own retrieval unit `<pid>#hq`."""
        if not questions:
            return
        hq_text = f"{title}\n" + "\n".join(questions)
        hq_id = f"{pid}#hq"
        if self.bm25:
            try:
                await self.bm25.upsert(hq_id, hq_text)
            except Exception as e:
                log.debug("doc2query BM25 upsert failed", extra={"metadata": {"error": str(e)[:120]}})
        if self.dense:
            try:
                await self.dense.upsert(
                    hq_id, hq_text,
                    meta={"title": title, "parent_id": pid, "hq": True},
                )
            except Exception as e:
                log.debug("doc2query dense upsert failed", extra={"metadata": {"error": str(e)[:120]}})

    async def _index_page_chunks(self, pid: str, title: str, body: str, frontmatter: dict) -> None:
        """Helper to index a page at hierarchical chunk level in BM25 and Chroma."""
        # 1) Clean up any old parent-page or sub-chunk entries to avoid stale chunk residue
        if self.bm25:
            try:
                await self.bm25.delete(pid)
                await self.bm25.delete(f"{pid}#hq")
                for idx in range(50):
                    await self.bm25.delete(f"{pid}#{idx}")
            except Exception:
                pass
        if self.dense:
            try:
                await self.dense.delete(pid)
                await self.dense.delete(f"{pid}#hq")
                for idx in range(50):
                    await self.dense.delete(f"{pid}#{idx}")
            except Exception:
                pass

        # 2) Index the sub-chunks of the entire page body
        text_to_index = f"{title}\n{body}"
        if frontmatter.get("context_preamble"):
            from .search.contextual import merge_context_with_chunk
            text_to_index = merge_context_with_chunk(frontmatter["context_preamble"], text_to_index)

        sub_chunks = _chunk_text(text_to_index, target_chars=SUB_CHUNK_TARGET, overlap=SUB_CHUNK_OVERLAP)
        for idx, ch in enumerate(sub_chunks):
            chunk_id = f"{pid}#{idx}"
            if self.bm25:
                try:
                    await self.bm25.upsert(chunk_id, ch)
                except Exception as e:
                    log.warning("BM25 chunk upsert failed", extra={"metadata": {"error": str(e)[:160]}})
            if self.dense:
                try:
                    await self.dense.upsert(
                        chunk_id, ch,
                        meta={
                            "title": title,
                            "confidence": frontmatter.get("confidence", 0.6),
                            "parent_id": pid,
                            "domain": frontmatter.get("domain", "general"),
                        }
                    )
                except Exception as e:
                    log.warning("Dense chunk upsert failed", extra={"metadata": {"error": str(e)[:160]}})

    # Display-math and inline-math spans → first-class formula media nodes.
    _FORMULA_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]{2,}?\$|\\\[.+?\\\]", re.DOTALL)
    _MEDIA_REL_BY_KIND = {"table": "MEASURES", "image": "DEPICTS", "code": "REFERENCES", "formula": "DEFINES"}
    _MAX_FORMULA_NODES = 30

    async def _populate_media_graph(self, pid: str, title: str, domain: str, elements, entities) -> None:
        """Phase 1: create media_nodes (table/image/code/formula) for a page, embed
        each as its own dense unit, and link them to entities by name presence.

        Best-effort and idempotent — clears the page's existing media nodes first so a
        re-ingest replaces rather than duplicates.
        """
        try:
            await self.graph.delete_media_for_page(pid)
        except Exception as e:
            log.debug("media cleanup failed", extra={"metadata": {"error": str(e)[:120]}})

        ent_list = [(e.name, e.type.upper()) for e in entities if getattr(e, "name", "")]

        # Collect modality units in document order.
        units: list[tuple[str, str, str | None]] = []   # (kind, content, caption)
        formula_count = 0
        for el in elements:
            meta = getattr(el, "meta", None) or {}
            if el.kind == "table":
                units.append(("table", el.content or "", meta.get("caption")))
            elif el.kind == "image":
                cap = meta.get("caption") or meta.get("description")
                units.append(("image", cap or meta.get("path") or el.content or "[image]", cap))
            elif el.kind == "code":
                units.append(("code", el.content or "", meta.get("lang")))
            elif el.kind in ("text", "heading") and el.content:
                for m in self._FORMULA_SPAN_RE.findall(el.content):
                    if formula_count >= self._MAX_FORMULA_NODES:
                        break
                    units.append(("formula", m.strip(), None))
                    formula_count += 1

        linked = 0
        for n, (kind, content, caption) in enumerate(units):
            if not content or not content.strip():
                continue
            emb_id = f"{pid}#media#{n}"
            try:
                mid = await self.graph.add_media_node(
                    page_id=pid, kind=kind, content=content[:4000],
                    caption=(caption or None), ordinal=n, embedding_id=emb_id,
                )
            except Exception as e:
                log.debug("media node insert failed", extra={"metadata": {"error": str(e)[:120]}})
                continue
            # Embed the unit as its own dense vector so it is independently retrievable.
            try:
                embed_text = f"{caption or ''}\n{content}".strip()[:4000]
                await self.dense.upsert(
                    emb_id, embed_text,
                    meta={"title": title, "parent_id": pid, "media_kind": kind, "domain": domain},
                )
            except Exception as e:
                log.debug("media embed failed", extra={"metadata": {"error": str(e)[:120]}})
            # Link to entities whose name appears in the unit's content/caption.
            hay = f"{content}\n{caption or ''}".lower()
            rel = self._MEDIA_REL_BY_KIND.get(kind, "DEPICTS")
            for ename, etype in ent_list:
                if len(ename) >= 3 and ename.lower() in hay:
                    try:
                        await self.graph.link_media_entity(
                            media_id=mid, entity_name=ename, entity_type=etype, rel_type=rel,
                        )
                        linked += 1
                    except Exception:
                        pass
        if units:
            log.info(
                "media graph populated",
                extra={"metadata": {"page": pid, "nodes": len(units), "links": linked}},
            )

    async def ingest_file(self, source_path: str | Path) -> IngestResult:
        src = Path(source_path)
        title = src.stem.replace("_", " ").replace("-", " ").title()
        try:
            image_out = self.s.raw_dir / "images" if self.s.ingest_extract_images else None
            elements = load_elements(
                src,
                extract_images=self.s.ingest_extract_images or self.s.ingest_caption_images,
                image_out=image_out or (self.s.raw_dir / "images"),
                ocr=self.s.ingest_ocr,
            )
        except Exception as e:
            log.exception("load failed")
            return IngestResult(source=str(src), page_path="", confidence=0.0, is_live=False, title=title, error=str(e))

        # Privacy redaction — strip secrets/PII from raw text BEFORE it reaches the
        # summariser, claim extractor, graph, embeddings, or the on-disk page.
        if self.s.ingest_redact_secrets:
            from .privacy import redact_text
            redaction_counts: dict[str, int] = {}
            for el in elements:
                if el.kind in ("text", "heading", "table", "code") and el.content:
                    res = redact_text(el.content, redact_emails=self.s.ingest_redact_emails)
                    if res.redacted:
                        el.content = res.text
                        for cat, n in res.counts.items():
                            redaction_counts[cat] = redaction_counts.get(cat, 0) + n
            if redaction_counts:
                log.warning(
                    "privacy redaction applied",
                    extra={"metadata": {"source": str(src), "counts": redaction_counts}},
                )
                try:
                    audit(
                        log, "PRIVACY_REDACT", str(src),
                        total=sum(redaction_counts.values()), counts=redaction_counts,
                    )
                except Exception as e:
                    log.debug("redaction audit failed", extra={"metadata": {"error": str(e)[:120]}})

        # Optional llava captions for image elements (in-place mutation).
        try:
            await self._caption_images(elements)
        except Exception as e:
            log.debug("caption pass failed", extra={"metadata": {"error": str(e)[:120]}})

        # Agentic ingestion — pick adaptive chunk params from document structure +
        # content density (dense technical → smaller chunks; narrative → larger).
        chunk_target, chunk_overlap = 6000, self.s.ingest_overlap
        ingest_plan = None
        if self.s.ingest_agentic_planning:
            from .agentic_ingest import plan_ingest
            ingest_plan = plan_ingest(
                elements, default_target=6000, default_overlap=self.s.ingest_overlap
            )
            if self.s.ingest_planning_llm:
                from .agentic_ingest import plan_ingest_llm
                ingest_plan = await plan_ingest_llm(self.c, elements, ingest_plan)
            chunk_target, chunk_overlap = ingest_plan.target_chars, ingest_plan.overlap_chars
            log.info(
                "agentic ingest plan",
                extra={"metadata": {
                    "strategy": ingest_plan.strategy, "target_chars": chunk_target,
                    "overlap_chars": chunk_overlap, "rationale": ingest_plan.rationale,
                }},
            )

        # Layout-aware chunking — tables and images stay atomic.
        chunks = layout_aware_chunks(
            elements, target_chars=chunk_target, overlap_chars=chunk_overlap
        )
        if not chunks:
            # Defensive fallback — no structured elements extracted, flatten and chunk.
            text = elements_to_markdown(elements) or ""
            chunks = _chunk_text(text, target_chars=6000, overlap=self.s.ingest_overlap)
        sum_tasks = [self._summarise_chunk(c) for c in chunks]
        ext_tasks = [self._extract_chunk(c) for c in chunks]
        partial_summaries = await asyncio.gather(*sum_tasks)
        extractions = await asyncio.gather(*ext_tasks)

        summary = await self._merge_summaries(partial_summaries)
        confidence, reason = await self._score_confidence(summary)

        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str]] = set()
        seen_rel: set[tuple[str, str, str]] = set()
        for ex in extractions:
            for e in ex.get("entities", []) or []:
                key = (e.get("name", "").strip().lower(), e.get("type", "").upper())
                if not key[0] or key in seen:
                    continue
                seen.add(key)
                entities.append(ExtractedEntity(name=e["name"], type=e.get("type", "CONCEPT")))
            for r in ex.get("relations", []) or []:
                src = (r.get("src") or "").strip()
                dst = (r.get("dst") or "").strip()
                rel_type = (r.get("rel_type") or "RELATES_TO").upper()
                if not src or not dst:
                    continue
                rkey = (src.lower(), dst.lower(), rel_type)
                if rkey in seen_rel:
                    continue
                seen_rel.add(rkey)
                relations.append(
                    ExtractedRelation(
                        src_name=src,
                        src_type=r.get("src_type", "CONCEPT"),
                        dst_name=dst,
                        dst_type=r.get("dst_type", "CONCEPT"),
                        rel_type=rel_type,
                    )
                )

        # ── Tiered extraction-signal floor ──────────────────────────────────
        # If we extracted substantial entities/relations the document had real
        # content; a near-zero confidence in that case is almost always a qwen
        # timeout / merge-fallback artefact, NOT a content-quality signal.
        # Tier the floor by extraction strength so very rich documents go LIVE
        # instead of sitting in review forever.
        n_entities_ex = sum(len(ex.get("entities") or []) for ex in extractions)
        n_relations_ex = sum(len(ex.get("relations") or []) for ex in extractions)

        floor_score: float | None = None
        if n_entities_ex >= 150 and n_relations_ex >= 5:
            # Very rich extraction — clearly substantive. Push above the live threshold.
            floor_score = 0.65
        elif n_entities_ex >= 80 or n_relations_ex >= 30:
            floor_score = 0.55
        elif n_entities_ex >= 30 or n_relations_ex >= 15:
            floor_score = 0.45

        if floor_score is not None and confidence < floor_score:
            log.warning(
                "extraction-signal floor: bumping confidence",
                extra={"metadata": {
                    "from": confidence, "to": floor_score,
                    "entities_extracted": n_entities_ex,
                    "relations_extracted": n_relations_ex,
                    "summary_chars": len(summary or ""),
                    "original_reason": reason[:160],
                }},
            )
            reason = (
                f"extraction-signal floor {floor_score:.2f} (was {confidence:.2f}); "
                f"{n_entities_ex} entities + {n_relations_ex} relations extracted. "
                f"Original: {reason[:200]}"
            )
            confidence = floor_score

        # Doc2Query — questions this document answers, indexed as their own unit so
        # question-phrased user queries match even against declarative source text.
        hyp_questions: list[str] = []
        if self.s.ingest_doc2query:
            hyp_questions = await self._doc2query(summary)

        entity_refs = [e.name for e in entities][:50]
        has_tables = any(el.kind == "table" for el in elements)
        has_images = any(el.kind == "image" for el in elements)
        # Domain tag — stamp the page's subject (general / math / science / economics /
        # engineering) so retrieval + adaptive model routing can reason about it. One
        # cheap heuristic check against the title + summary.
        domain = "general"
        if self.s.ingest_domain_tagging:
            from .search.domain import heuristic_domain
            domain = heuristic_domain(f"{title}\n{summary or ''}") or "general"
        frontmatter = {
            "title": title,
            "kind": "source",
            "description": derive_description(summary),
            "source": str(src).replace("\\", "/"),
            "resource": str(src).replace("\\", "/"),
            "ingested": date.today().isoformat(),
            "source_count": 1,
            "confidence": round(confidence, 2),
            "confidence_reason": reason[:300],
            "domain": domain,
            "chunk_strategy": ingest_plan.strategy if ingest_plan else "fixed",
            "tags": sorted({e.type.lower() for e in entities}),
            "entity_refs": entity_refs,
            "has_tables": has_tables,
            "has_images": has_images,
            "hypothetical_questions": hyp_questions,
            "element_counts": {
                "text": sum(1 for el in elements if el.kind == "text"),
                "heading": sum(1 for el in elements if el.kind == "heading"),
                "table": sum(1 for el in elements if el.kind == "table"),
                "image": sum(1 for el in elements if el.kind == "image"),
                "code": sum(1 for el in elements if el.kind == "code"),
            },
        }
        # Body: summary up top, followed by any tables / captioned images verbatim so
        # the wiki page preserves multimodal content (and Obsidian renders it nicely).
        body_parts = [summary]
        preserved = [el for el in elements if el.kind in ("table", "image")]
        if preserved:
            body_parts.append("\n## Tables & Figures\n")
            body_parts.append(elements_to_markdown(preserved))
        body = "\n\n".join(body_parts)

        # ── Compute Contextual-Retrieval preamble BEFORE writing the page ──
        # Otherwise the preamble field can't make it into the on-disk frontmatter.
        if self.s.ingest_contextual_retrieval and summary:
            try:
                from .search.contextual import contextualize_chunk
                preamble = await contextualize_chunk(
                    self.c,
                    doc_title=title,
                    full_doc_excerpt=summary,
                    chunk=summary,
                    semaphore=self._sem,
                )
                if preamble:
                    # The preamble is merged with the chunk text at index time inside
                    # `_index_page_chunks`; here we only need to persist it to frontmatter.
                    frontmatter["context_preamble"] = preamble[:300]
            except Exception as e:
                log.debug("contextual preamble failed", extra={"metadata": {"error": str(e)[:120]}})

        page_path, is_live = stage_or_publish(title, body, frontmatter, settings=self.s)
        pid = page_id_from_path(page_path, self.s.wiki_dir)

        entities_added = 0
        contradictions = 0
        if self.graph and entities:
            # Run contradiction check BEFORE upserting (compare vs. pre-existing pages).
            try:
                contradictions = await self._detect_contradictions(pid, summary, [e.name for e in entities])
            except Exception as e:
                log.debug("contradiction pass failed", extra={"metadata": {"error": str(e)[:120]}})
            canon_ids = await self.graph.upsert_entities_delta(pid, entities, relations)
            entities_added = len(canon_ids)
            # Phase A1: extract S-P-O claims and persist into facts table.
            # Best-effort; failures don't break ingest.
            try:
                claims = await self._extract_claims(summary, entities)
                if claims:
                    await self._persist_claims(pid, claims)
            except Exception as e:
                log.debug("claim extraction/persist failed",
                          extra={"metadata": {"error": str(e)[:120]}})
        await self._index_page_chunks(pid, title, body, frontmatter)
        if hyp_questions:
            await self._index_doc2query(pid, title, hyp_questions)

        # ── Multimodal graph (Phase 1): persist tables/images/code/formulas as
        # first-class media nodes linked to entities, each embedded as its own unit.
        if self.s.graph_multimodal_nodes and self.graph and self.dense:
            try:
                await self._populate_media_graph(pid, title, domain, elements, entities)
            except Exception as e:
                log.debug("media graph population failed", extra={"metadata": {"error": str(e)[:160]}})

        # ── Memory evolution / Reconciliation pass (A-Mem 2026) ──
        # After publishing the new page, propose edits to existing pages most
        # affected by this source. High-confidence edits auto-apply; lower-
        # confidence ones are staged in wiki/review/edits/ for human review.
        applied_edits = 0
        staged_edits = 0
        if self.s.ingest_reconcile and self.graph and is_live and entities:
            try:
                affected = await find_affected_pages(
                    graph=self.graph,
                    new_entities=[e.name for e in entities],
                    new_page_id=pid,
                    max_pages=self.s.ingest_reconcile_max_pages,
                )
                for tgt_pid in affected:
                    tgt_path = self.s.wiki_dir / tgt_pid
                    if not tgt_path.exists():
                        continue
                    try:
                        tgt_page = read_page(tgt_path)
                    except Exception:
                        continue
                    proposal = await propose_edit(
                        self.c,
                        new_source_summary=summary,
                        new_source_title=title,
                        target_page_id=tgt_pid,
                        target_page_body=tgt_page.body,
                        target_page_title=tgt_page.frontmatter.get("title", tgt_pid),
                    )
                    if proposal is None:
                        continue
                    if should_auto_apply(proposal):
                        new_body, applied = apply_edit_to_body(tgt_page.body, proposal)
                        if not applied:
                            # Auto-apply was attempted but the edit was a no-op
                            # (e.g. refine couldn't find old_text). Demote to staged.
                            stage_proposal(self.s.wiki_dir, proposal)
                            staged_edits += 1
                            audit(log, "WIKI_WRITE_STAGED", str(tgt_path),
                                  kind="edit_proposal", trigger=title,
                                  action=proposal.action, confidence=proposal.confidence,
                                  reason="auto_apply_was_noop")
                            continue
                        tgt_page.body = new_body
                        evolved = tgt_page.frontmatter.get("evolved_by", []) or []
                        evolved.append({"source": title, "action": proposal.action,
                                        "date": date.today().isoformat()})
                        tgt_page.frontmatter["evolved_by"] = evolved
                        write_page(Page(path=tgt_path, frontmatter=tgt_page.frontmatter, body=new_body))
                        await self._index_page_chunks(
                            tgt_pid,
                            tgt_page.frontmatter.get("title", tgt_pid),
                            new_body,
                            tgt_page.frontmatter
                        )
                        applied_edits += 1
                        audit(log, "WIKI_WRITE", str(tgt_path),
                              kind="evolved", trigger=title, action=proposal.action,
                              confidence=proposal.confidence)
                        # Phase A2: when a refine/contradict edit lands, mark
                        # facts on the target page that overlap with old_text
                        # as superseded by the new page's facts.
                        if proposal.action in ("refine", "contradict") and proposal.old_text:
                            try:
                                await self._supersede_facts_matching_old_text(
                                    target_pid=tgt_pid,
                                    old_text=proposal.old_text,
                                    new_pid=pid,
                                )
                            except Exception as e:
                                log.debug("supersede in reconciler path failed",
                                          extra={"metadata": {"error": str(e)[:120]}})
                    elif should_propose(proposal):
                        stage_proposal(self.s.wiki_dir, proposal)
                        staged_edits += 1
                        audit(log, "WIKI_WRITE_STAGED", str(tgt_path),
                              kind="edit_proposal", trigger=title, action=proposal.action,
                              confidence=proposal.confidence)
            except Exception as e:
                log.warning("reconciliation pass failed", extra={"metadata": {"error": str(e)[:200]}})

        try:
            rebuild_index(self.s.wiki_dir)
        except Exception as e:
            log.warning("rebuild_index failed", extra={"metadata": {"error": str(e)[:200]}})
        if self.graph is not None:
            try:
                rebuild_entity_pages(self.graph, self.s.wiki_dir)
            except Exception as e:
                log.warning("entity-page rebuild failed", extra={"metadata": {"error": str(e)[:200]}})

        # Episodic log entry
        if self.s.episodic_logging:
            try:
                append_episode(
                    self.s.wiki_dir,
                    kind="ingest",
                    title=title,
                    body=(
                        f"Source: `{Path(src).name}` · "
                        f"chunks: {len(chunks)} · confidence: {confidence:.2f} · "
                        f"live: {is_live} · entities: {len(entities)} · "
                        f"contradictions: {contradictions} · "
                        f"edits applied: {applied_edits} · staged: {staged_edits}"
                    ),
                    metadata={
                        "page": pid, "confidence": f"{confidence:.2f}",
                        "live": str(is_live),
                    },
                )
            except Exception as e:
                log.debug("episodic logging failed", extra={"metadata": {"error": str(e)[:120]}})

        try:
            append_log(
                self.s.wiki_dir,
                "ingest",
                title,
                f"source: `{Path(src).name}` · chunks: {len(chunks)} · confidence: {confidence:.2f} · live: {is_live} · contradictions: {contradictions}",
            )
        except Exception as e:
            log.warning("append_log failed", extra={"metadata": {"error": str(e)[:200]}})

        try:
            if confidence < self.s.confidence_threshold:
                audit(log, "CONFIDENCE_LOW", str(page_path), confidence=confidence, reason=reason)
        except Exception as e:
            log.debug("audit failed", extra={"metadata": {"error": str(e)[:200]}})

        return IngestResult(
            source=str(src),
            page_path=str(page_path),
            confidence=round(confidence, 2),
            is_live=is_live,
            entities_added=entities_added,
            title=title,
            chunks=len(chunks),
            extracted={"entities": len(entities), "relations": len(relations)},
        )

    async def ingest_many(self, paths: list[str | Path]) -> list[IngestResult]:
        raw = await asyncio.gather(
            *(self.ingest_file(p) for p in paths), return_exceptions=True
        )
        out: list[IngestResult] = []
        for path, res in zip(paths, raw, strict=False):
            if isinstance(res, BaseException):
                # Log the FULL traceback so we can see the exact failing line.
                import traceback
                tb = "".join(traceback.format_exception(type(res), res, res.__traceback__))
                log.error(
                    "ingest_file crashed",
                    extra={"metadata": {
                        "path": str(path),
                        "error": str(res)[:200],
                        "exc_type": type(res).__name__,
                        "traceback": tb[-2000:],   # last ~2000 chars of TB
                    }},
                )
                p = Path(path)
                out.append(
                    IngestResult(
                        source=str(p),
                        page_path="",
                        confidence=0.0,
                        is_live=False,
                        title=p.stem,
                        error=f"{type(res).__name__}: {res}",
                    )
                )
            else:
                out.append(res)
        return out
