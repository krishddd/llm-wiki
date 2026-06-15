"""Agentic-layer settings. Loaded from env with the `AGENTIC_` prefix.

Kept separate from `src.config.Settings` so the agentic layer can be enabled,
tuned, or disabled without touching the core engine config.
"""
from __future__ import annotations

from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover
    from pydantic import BaseSettings  # type: ignore[no-redef]
    SettingsConfigDict = None  # type: ignore[assignment,misc]
    _HAS_PYDANTIC_SETTINGS = False


class AgenticSettings(BaseSettings):
    agentic_enabled: bool = True
    agentic_max_iterations: int = 3
    agentic_coverage_threshold: float = 0.85
    agentic_quick_draft_model: str = "gemma"        # client method name
    agentic_sca_model: str = "qwen"                 # client method name
    agentic_complexity_threshold: str = "multi_hop" # min intent to trigger agentic path
    agentic_fanout_concurrency: int = 2
    agentic_per_iteration_top_k: int = 5
    agentic_draft_max_chars: int = 1200

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:  # pragma: no cover
        class Config:
            env_file = ".env"
            extra = "ignore"


@lru_cache(maxsize=1)
def get_agentic_settings() -> AgenticSettings:
    return AgenticSettings()
