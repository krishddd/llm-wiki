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
    # is a no-op (we do NOT have llama3.2 installed; see llm.qwen()).
    model_embed: str = "nomic-embed-text:latest"
    model_reason: str = "qwen3:14b"
    model_summary: str = "gemma4:e4b"
    model_fast: str = "qwen3:14b"
    model_vision: str = "llava:7b"

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

    # Adaptive retrieval — classify question intent (factual / multi_hop /
    # synthesis / exhaustive) and pick top_k + full_page_mode + graph_expand
    # accordingly. Heuristic first, gemma fallback on ambiguous cases.
    query_adaptive_retrieval: bool = True

    # Reflection / critique pass — post-synth gemma critique; optionally triggers
    # a single re-synthesis when the draft is incomplete or under-cited.
    query_reflect: bool = True
    query_reflect_refine: bool = True   # actually re-synthesize on weak drafts

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
    episodic_retention_days: int = 14

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
