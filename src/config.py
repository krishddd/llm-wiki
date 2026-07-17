"""Central settings. Reads .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow `model_*` field names without pydantic warnings
    )

    ollama_host: str = "http://localhost:11434"

    # 4-model stack. model_fast == model_reason so the qwen-timeout fallback path
    # is a no-op (we do NOT have llama3.2 installed; see llm.reason()).
    model_embed: str = "nomic-embed-text:latest"
    # Domain-specialized embedding — a stronger embedder for STEM / notation-heavy
    # content (maths, science, economics, engineering), where the small general model
    # loses recall. When `embed_stem_enabled`, quantitative pages are ALSO indexed into
    # a separate STEM dense collection and quantitative queries are routed to it.
    # Off by default (needs the model pulled + an index rebuild). bge-m3 is a strong,
    # Ollama-available default; swap for any embedder you prefer.
    model_embed_stem: str = "bge-m3"
    embed_stem_enabled: bool = False
    model_reason: str = "qwen3:14b"
    model_summary: str = "gemma4:e4b"
    model_fast: str = "qwen3:14b"
    model_vision: str = "llava:7b"

    # ── v4/v6 multi-provider LLM fleet ───────────────────────────────────────────
    # Each text role can be routed to a hosted, OpenAI-compatible provider instead of
    # local Ollama. Default "ollama" for every role → no behaviour change. A role
    # silently falls back to Ollama if its provider's API key or model is unset.
    # Providers: "ollama" | "groq" | "github" | "gemini" | "openai" | "anthropic"
    #            | "xai" | "openrouter" | "custom" (any OpenAI-compatible gateway).
    provider_summary: str = "ollama"   # summary role (summarise / extract)
    provider_reason: str = "ollama"    # reason role (synthesis / deep reasoning)
    provider_fast: str = "ollama"      # fast-agent role
    provider_solver: str = "ollama"    # VibeThinker reasoning role
    provider_embed: str = "ollama"     # embeddings ("ollama" | "gemini" | "openai" | "custom")
    provider_vision: str = "ollama"    # image captioning (llava role)

    # Provider API keys — SECRETS, supplied via env / .env ONLY (never commit real values).
    groq_api_key: str = ""
    github_models_token: str = ""
    google_genai_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    openrouter_api_key: str = ""
    custom_api_key: str = ""           # optional — local gateways (vLLM/LM Studio) need none

    # Provider model names.
    groq_model: str = "llama-3.3-70b-versatile"
    github_models_model: str = "openai/gpt-4.1-mini"
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embed_model: str = "text-embedding-004"
    openai_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"
    anthropic_model: str = "claude-sonnet-5"
    xai_model: str = "grok-4"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    custom_model: str = ""             # e.g. whatever your vLLM/LM Studio serves
    custom_embed_model: str = ""
    # Vision (multimodal) model names. Most flagship chat models are already
    # multimodal; empty → vision falls back to Ollama for that provider.
    groq_vision_model: str = ""
    github_models_vision_model: str = "openai/gpt-4.1-mini"
    gemini_vision_model: str = "gemini-2.5-flash-lite"
    openai_vision_model: str = "gpt-4.1-mini"
    anthropic_vision_model: str = "claude-sonnet-5"
    xai_vision_model: str = ""
    openrouter_vision_model: str = ""
    custom_vision_model: str = ""

    # Provider base URLs (OpenAI-compatible gateways; override if a provider moves).
    groq_base_url: str = "https://api.groq.com/openai/v1"
    github_models_base_url: str = "https://models.github.ai/inference"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"   # OpenAI-compat endpoint
    xai_base_url: str = "https://api.x.ai/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    custom_base_url: str = ""          # e.g. http://localhost:8001/v1 (vLLM), http://localhost:1234/v1 (LM Studio)

    # Reasoning specialist (VibeThinker — AIME-class maths / STEM / code). Used for
    # the *adaptive routing* path only: quantitative questions reason here, then qwen
    # formats + cites the result. Empty string disables routing entirely.
    # Serve via Ollama (GGUF) or a vLLM sidecar; recommended sampler temp≈0.6 top_p≈0.95.
    model_solver: str = "vibethinker:3b"

    confidence_threshold: float = 0.60
    pdf_chunk_pages: int = 3
    ingest_overlap: int = 200
    # Strictly sequential — avoids Ollama model-swap thrashing.
    max_concurrent_llm_req: int = 1

    # Multimodal ingest flags (opt-in because they add LLM calls / optional deps).
    ingest_caption_images: bool = False  # True → run llava on each extracted image
    ingest_ocr: bool = True               # True → pytesseract OCR on scanned PDF pages
    ingest_extract_images: bool = False   # True → save embedded images to wiki/raw/images/
    ingest_max_image_caption: int = 20    # cap per-document image captions to avoid LLM blowup

    # Advanced RAG flags.
    # Anthropic Contextual Retrieval — adds 1 gemma call per chunk at ingest time
    # (huge recall win, ~30% reduction in retrieval failures, but costs LLM calls).
    ingest_contextual_retrieval: bool = True

    # CRAG-style relevance evaluation BEFORE synthesis — adds 1 gemma call per
    # retrieved page at query time. Filters off-topic retrievals that would
    # otherwise corrupt the synthesis. Cheap & high-value.
    query_relevance_eval: bool = True

    # RAG-Fusion / multi-query — generates 2-3 paraphrases of each (sub-)query
    # and RRF-fuses results. One extra qwen call per query.
    query_multi_query: bool = True

    # Adaptive model routing — detect quantitative/STEM questions (maths, economics,
    # science, engineering/industrial materials) and route the *reasoning* to
    # `model_solver` (VibeThinker), then let qwen format + cite. Self-disables if
    # `model_solver` is empty or not installed (falls back to qwen synthesis).
    route_solver_enabled: bool = True
    # Allow a gemma LLM fallback for domain detection when regex signals are absent.
    route_solver_llm_fallback: bool = False
    # VibeThinker sampler — authors recommend temperature 0.6, top_p 0.95.
    solver_temperature: float = 0.6
    # Domain tagging at ingest — stamp pages with a `domain:` frontmatter field so
    # retrieval/routing can reason about subject matter. One heuristic check per page.
    ingest_domain_tagging: bool = True

    # Agentic ingestion — inspect each document's structure + content density and pick
    # adaptive chunk size/overlap (dense technical → smaller, narrative → larger) instead
    # of a fixed 6000-char target. Heuristic is cheap and on by default; the gemma
    # refinement (one extra call per doc) is opt-in.
    ingest_agentic_planning: bool = True
    ingest_planning_llm: bool = False

    # Multimodal knowledge graph (Phase 1) — persist tables/images/code/formulas as
    # first-class media_nodes linked to entities, each embedded as its own dense unit.
    # Off by default (populating requires re-ingest). Retrieval through media nodes is
    # a later phase; Phase 1 is data-only.
    graph_multimodal_nodes: bool = False

    # Privacy / secret redaction — strip API keys, JWTs, private keys and plaintext
    # passwords from raw source text BEFORE it reaches the summariser / claims / graph /
    # embeddings / on-disk page. Audit-logged as PRIVACY_REDACT. (CLAUDE.md policy.)
    ingest_redact_secrets: bool = True
    # Emails are PII but public author emails are legitimate content — opt-in only.
    ingest_redact_emails: bool = False

    # Adaptive retrieval — classify question intent (factual / multi_hop /
    # synthesis / exhaustive) and pick top_k + full_page_mode + graph_expand
    # accordingly. Heuristic first, gemma fallback on ambiguous cases.
    query_adaptive_retrieval: bool = True

    # Reflection / critique pass — post-synth gemma critique; optionally triggers
    # a single re-synthesis when the draft is incomplete or under-cited.
    query_reflect: bool = True
    query_reflect_refine: bool = True   # actually re-synthesize on weak drafts

    # ── Best-of-best RAG package ─────────────────────────────────────────────
    # Small-to-big retrieval — rerank/synthesise on the sub-chunks that actually
    # matched (plus neighbours) instead of the first 4000 chars of the parent page.
    query_chunk_context: bool = True
    # Anti-feedback-loop — score multiplier (<1) for machine-written pages
    # (synthesis/promoted/crystallized) so save-backs never outrank primary sources.
    # Set 1.0 to disable.
    retrieval_synth_downweight: float = 0.85
    # Lost-in-the-middle mitigation — ends-load the synthesis context (best page
    # first, runner-up last) to counter positional attention decay.
    query_litm_reorder: bool = True
    # NLI-lite claim verification — ONE batched gemma call checks each cited claim
    # sentence against its cited snippet; unsupported claims drag confidence down.
    query_claim_verify: bool = True
    # Doc2Query — at ingest, generate the questions each document answers and index
    # them as their own retrieval unit (`<pid>#hq`). One gemma call per document.
    ingest_doc2query: bool = True
    # RAPTOR-lite topic pages — weekly clustering of live pages into topic overviews
    # so corpus-level ("what are the main themes…") questions have a retrievable page.
    job_build_topics_enabled: bool = True
    topics_min_cluster: int = 3
    topics_max: int = 12
    topics_sim_threshold: float = 0.62

    # ── Review Autopilot ─────────────────────────────────────────────────────
    # Evidence-grounded auto-verification of staged review pages: an LLM judge
    # re-reads each staged page AGAINST its original source (+ a deterministic
    # entity-grounding cross-check). composite ≥ accept → auto-publish;
    # ≤ reject → auto-archive (never deleted); in between → annotated for human
    # review. Runs inline after ingest for freshly staged pages, plus a daily
    # sweep job for the backlog.
    review_autopilot_enabled: bool = True        # inline post-ingest pass
    job_review_autopilot_enabled: bool = True    # daily backlog sweep (04:30 UTC)
    review_accept_threshold: float = 0.70
    review_reject_threshold: float = 0.30
    # Borderline composites get a second judge (reason role) and the votes average.
    review_second_opinion: bool = True
    review_autopilot_max_pages: int = 25         # per sweep, bounds LLM cost

    # 2026 features:
    # Memory evolution — A-Mem reconciler. After a new source is ingested, edit
    # affected pre-existing pages instead of leaving them frozen.
    ingest_reconcile: bool = True
    ingest_reconcile_max_pages: int = 5

    # Episodic log — append every query/ingest event to wiki/episodic/<date>.md
    episodic_logging: bool = True
    episodic_retention_days: int = 14

    # Per-claim confidence — synth model emits [Page]^0.NN markers; we parse + display
    query_per_claim_confidence: bool = True

    # Phase B — Memory lifecycle (Ebbinghaus decay + reinforcement)
    lifecycle_enabled: bool = True
    decay_half_life_days: float = 90.0
    reinforcement_threshold: int = 3
    reinforcement_window_days: int = 14

    # Phase D — APScheduler-driven background jobs
    scheduler_enabled: bool = True
    job_decay_sweep_enabled: bool = True
    job_episodic_prune_enabled: bool = True
    job_promote_episodic_enabled: bool = True
    job_lint_autofix_enabled: bool = True
    job_detect_procedures_enabled: bool = True
    job_page_compaction_enabled: bool = True

    wiki_dir: Path = Path("wiki")
    raw_dir: Path = Path("wiki/raw")
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Generous — cold-load of gemma4:e4b or qwen3:14b can take 30-60s.
    llm_timeout: float = 600.0
    llm_fast_timeout: float = 120.0

    def required_models(self) -> list[str]:
        # Deduplicate — model_fast may == model_reason in the 4-model stack.
        seen: list[str] = []
        for m in (self.model_embed, self.model_reason, self.model_summary, self.model_fast, self.model_vision):
            if m and m not in seen:
                seen.append(m)
        return seen


@lru_cache
def get_settings() -> Settings:
    return Settings()
