# Design Proposal: Unified Multimodal Knowledge Graph

> Status: **Proposal — awaiting approval before implementation.**
> Author: agent-assisted. Companion to `CLAUDE.md` (schema authority).
> Scope: make tables / images / code / formulas first-class graph nodes, linked to
> the entities and pages they belong to, retrievable on their own.

---

## 1. Problem

Today, multimodal content is **invisible to the knowledge graph**. Tables, images,
and code are parsed *on the fly* from page Markdown by `search/multimodal.py`
(`extract_excerpts`, kinds `table | image | code`) and attached to a `Citation` as an
ephemeral side-channel. They are re-parsed on every query and never persisted as
addressable units.

Consequences:
- A stress-strain **table** is not connected to the entity `6061-T6` — asking about
  the alloy won't surface the table unless the table's text lexically matches.
- A **figure** can't be retrieved on its own; it only rides along with its page.
- Modality units are **not embedded as their own units** — a table is buried inside a
  ~1500-char page chunk, diluting its vector.
- They cannot participate in **2-hop graph expansion**.

This is the gap that 2026 multimodal-graph RAG (VimRAG, G²-Reader) closes by making
modality units typed nodes with semantic + spatial edges.

## 2. Current state (grounded in the code)

| Component | What exists |
|---|---|
| `graph.py` | `entities`, `relations` (bi-temporal), `page_entities` (page↔entity), `facts` (bi-temporal S-P-O). 2-hop expansion at retrieval. |
| `search/multimodal.py` | `extract_excerpts()` parses table/image/code from page body at query time; `MultimodalExcerpt`. Ephemeral. |
| `ingest.py` | Loads `DocElement`s (kinds `heading|text|table|image|code`); tables/images preserved verbatim in page body; images optionally captioned by llava. |
| `query.py` | `_build_context` re-extracts excerpts and attaches them to `Citation.excerpts`. |

## 3. Proposed schema (additive — new tables only)

```sql
-- A modality unit extracted from a page: table, image, code, or formula.
CREATE TABLE IF NOT EXISTS media_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    kind TEXT NOT NULL,            -- table | image | code | formula
    ordinal INTEGER,              -- position within the page (adjacency)
    content TEXT NOT NULL,        -- table markdown / image caption+path / code / LaTeX
    caption TEXT,                 -- nearby caption or llava description
    embedding_id TEXT,            -- dense-index id, e.g. "<pid>#media#<n>"
    bbox TEXT,                    -- optional PDF spatial info (page, x0,y0,x1,y1)
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_media_page ON media_nodes(page_id);

-- Edges: modality node <-> entity (canonical id), reusing entity canonicalization.
CREATE TABLE IF NOT EXISTS media_entities (
    media_id  INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    rel_type  TEXT DEFAULT 'DEPICTS',   -- DEPICTS | MEASURES | DEFINES | REFERENCES
    PRIMARY KEY (media_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON media_entities(entity_id);
```

Migration follows the existing `_migrate_relations_table` pattern: `CREATE TABLE IF NOT
EXISTS` + idempotent guards. **Zero impact on existing DBs.**

## 4. Ingest changes

After element load + privacy redaction (and after entity extraction so canonical ids
exist):
1. Collect media `DocElement`s (`table`, `image`, `code`) **+** formula spans pulled
   from text via the existing `domain._FORMULA_RE`.
2. For each: insert a `media_nodes` row; embed its content (table markdown / image
   caption / code / LaTeX) into the dense index as its **own unit** (`<pid>#media#<n>`).
3. Link it to entities found in its caption / adjacent text span via `media_entities`,
   reusing `KnowledgeGraph` canonicalization (threshold 95). Pick `rel_type` by kind:
   table→`MEASURES`, image→`DEPICTS`, code→`REFERENCES`, formula→`DEFINES`.

New `KnowledgeGraph` methods: `add_media_node(...)`, `link_media_entity(...)`,
`media_for_entity(entity_name)`, `media_for_page(page_id)`.

## 5. Retrieval changes

- **Independent recall**: media units now have their own dense ids, so a relevant
  table/figure surfaces directly from hybrid search.
- **2-hop expansion**: when an entity is retrieved, also pull its linked
  `media_nodes` (`media_entities` join) — asking about `6061-T6` surfaces the stress
  table even if the table text didn't lexically match. Hooks into the existing graph
  expansion in `hybrid_search`.
- **Typed citation nodes**: extend `Citation` / `CitationExcerpt` to carry a
  `node_id` + linked entities, so the synthesizer can cite `[Table 3: 6061-T6
  properties]` as a first-class source rather than an attachment.

## 6. Edges & the "graph" (2026 framing)

Beyond entity links, add lightweight structural edges:
- `BELONGS_TO` — media → page (implicit via `page_id`).
- `NEAR` — media ↔ adjacent text span (caption/explanation), via `ordinal` (Markdown)
  or `bbox` (PDF). Gives the synthesizer the explanation that goes with a table.
- `MEASURES`/`DEPICTS`/`DEFINES`/`REFERENCES` — media → entity (above).

## 7. Lifecycle / bi-temporal consistency

Media nodes are immutable artifacts — no supersession. They inherit their page's
lifecycle: when a page is archived by lint auto-fix or decays out, its media nodes are
treated inactive (cascade on `page_id`). No new bi-temporal columns required.

## 8. Phasing (incremental, each independently shippable)

| Phase | Deliverable | Blast radius | Verifiable here? |
|---|---|---|---|
| **1** | Schema + ingest population + media embedding. **No retrieval change.** | `graph.py`, `ingest.py` | ✅ unit tests on in-memory sqlite |
| **2** | 2-hop expansion pulls linked media; media as citation nodes. | `hybrid.py`, `query.py` | ⚠️ partial (retrieval needs Chroma) |
| **3** | `NEAR` spatial edges, formula nodes, image-caption entity links. | ingest + loaders | ⚠️ partial |

**Recommendation: start with Phase 1** — it's pure data (additive tables + population),
isolated, and fully unit-testable with an in-memory graph, mirroring how the prior five
features were verified. Phases 2–3 land once Phase 1 is in and CI is green.

## 9. Flags

```python
graph_multimodal_nodes: bool = False   # default off — populating needs a re-ingest
```
Off → behaviour identical to today. On → ingest populates media tables and (Phase 2+)
retrieval expands through them.

## 10. Risks

- **Embedding cost**: +1 embed per media element at ingest (bounded by element count).
- **DB growth**: one row per table/image/code/formula.
- **Blast radius**: Phase 1 touches only `graph.py` + `ingest.py` (data only) — lowest
  risk; the riskier retrieval rewiring is deferred to Phase 2 behind the same flag.

## 11. Verification plan

- Unit tests (Phase 1): media-node insert, entity linking + canonicalization, the
  `media_for_entity` join — all on in-memory sqlite (no Ollama/Chroma needed).
- CI: `ruff` (blocking) + mocked unit tests. Integration (Chroma + Ollama) runs
  locally / in the integration workflow.

---

### Open questions for sign-off
1. **Formula nodes** — pull `$…$` / `$$…$$` spans as their own `formula` media nodes,
   or keep them inline in text? (Proposal: yes, as nodes — they're the core of the
   maths/STEM goal.)
2. **Image embedding** — embed the llava caption only (cheap, text-space, default), or
   add true image embeddings later (needs a vision embedder)? (Proposal: caption-only
   for Phase 1.)
3. **Phase 1 scope confirmation** — ship schema + population first with no retrieval
   change, as recommended?
