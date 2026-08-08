"""Runtime-enforced profile / schema contract for wiki pages.

Inspired by the Configurable Lifecycle Profiles idea (atomicstrata/llm-wiki-compiler):
the wiki's conventions — which `kind`s exist, which frontmatter fields are required,
what `domain`/type values are allowed, the entity/relation vocabularies — live in
`CLAUDE.md` as *prose*. Prose isn't enforced. This module turns them into a declarative,
**runtime-validated contract** applied at the page write surface, so a malformed page is
caught deterministically instead of drifting into the corpus.

Design:
- A `Profile` is a plain data object with a `DEFAULT_PROFILE` that matches the current
  schema exactly, so nothing is rejected that the pipeline already produces.
- Optional override: a JSON file (env `PROFILE_PATH`, or `profile.json` in the wiki dir).
  Only the keys present in the file override the defaults — partial profiles are fine.
- Enforcement mode (`PROFILE_ENFORCEMENT`): `off` | `warn` (default) | `strict`.
  `warn` logs + audits violations but still writes; `strict` raises `ProfileViolation`
  so the caller can route the page to review instead of publishing garbage.
- Validation is pure (no LLM, no I/O beyond loading the profile) and returns structured
  errors (hard) + warnings (soft) so both the writer and a corpus-audit endpoint reuse it.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class ProfileViolation(Exception):
    """Raised by `enforce()` in strict mode when a page violates the contract."""

    def __init__(self, page: str, errors: list[str]):
        self.page = page
        self.errors = errors
        super().__init__(f"{page}: " + "; ".join(errors))


@dataclass
class Profile:
    """The declarative contract. All sets are matched case-insensitively where noted."""
    # Allowed page `kind` values (the internal taxonomy).
    page_kinds: set[str] = field(default_factory=lambda: {
        "source", "entity", "synthesis", "promoted", "crystallized", "procedure",
        "topic", "episodic",
    })
    # Frontmatter fields every concept page MUST carry.
    required_fields: list[str] = field(default_factory=lambda: ["title", "kind"])
    # Allowed `domain` values (adaptive routing vocabulary). Empty domain is allowed.
    domains: set[str] = field(default_factory=lambda: {
        "general", "math", "science", "economics", "engineering",
    })
    # Fields that, when present, must be a list.
    list_fields: set[str] = field(default_factory=lambda: {"tags", "entity_refs"})
    # Knowledge-graph vocabularies (mirror src/graph.py so the whole contract is in one place).
    entity_types: set[str] = field(default_factory=lambda: {
        "PERSON", "ORG", "CONCEPT", "PLACE", "EVENT",
    })
    relation_types: set[str] = field(default_factory=lambda: {
        "RELATES_TO", "PART_OF", "CONTRADICTS", "SUPPORTS", "AUTHORED_BY", "OCCURRED_IN",
    })
    # Confidence must fall in this closed interval when present.
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    # Optional declared lifecycle states (informational; directory placement is the
    # real state machine — sources/ = live, review/ = staged, archive/ = retired).
    lifecycle_states: list[str] = field(default_factory=lambda: ["live", "review", "archived"])

    def as_dict(self) -> dict:
        return {
            "page_kinds": sorted(self.page_kinds),
            "required_fields": list(self.required_fields),
            "domains": sorted(self.domains),
            "list_fields": sorted(self.list_fields),
            "entity_types": sorted(self.entity_types),
            "relation_types": sorted(self.relation_types),
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "lifecycle_states": list(self.lifecycle_states),
        }


DEFAULT_PROFILE = Profile()

_SET_KEYS = {"page_kinds", "domains", "list_fields", "entity_types", "relation_types"}
_LIST_KEYS = {"required_fields", "lifecycle_states"}


def load_profile(path: str | Path | None) -> Profile:
    """Load a profile from JSON, overriding only the keys present. Returns DEFAULT on
    missing path or any parse error (fail-safe — a broken profile never breaks writes)."""
    if not path:
        return DEFAULT_PROFILE
    p = Path(path)
    if not p.exists():
        return DEFAULT_PROFILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("profile load failed, using default", extra={"metadata": {"path": str(p), "error": str(e)[:160]}})
        return DEFAULT_PROFILE
    prof = Profile()
    for k, v in (data or {}).items():
        if not hasattr(prof, k):
            continue
        if k in _SET_KEYS and isinstance(v, list):
            setattr(prof, k, {str(x) for x in v})
        elif k in _LIST_KEYS and isinstance(v, list):
            setattr(prof, k, [str(x) for x in v])
        elif k in ("confidence_min", "confidence_max"):
            with contextlib.suppress(TypeError, ValueError):
                setattr(prof, k, float(v))
    return prof


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)     # hard violations
    warnings: list[str] = field(default_factory=list)   # soft (recommended) issues


def validate_page(frontmatter: dict, *, profile: Profile = DEFAULT_PROFILE) -> ValidationResult:
    """Validate a page's frontmatter against the contract. Pure; never raises."""
    fm = frontmatter or {}
    errors: list[str] = []
    warnings: list[str] = []

    for req in profile.required_fields:
        if not fm.get(req):
            errors.append(f"missing required field '{req}'")

    kind = str(fm.get("kind", "")).strip().lower()
    if kind and kind not in profile.page_kinds:
        errors.append(f"kind '{kind}' not in profile ({', '.join(sorted(profile.page_kinds))})")

    if "confidence" in fm and fm.get("confidence") is not None:
        try:
            c = float(fm["confidence"])
            if not (profile.confidence_min <= c <= profile.confidence_max):
                errors.append(f"confidence {c} outside [{profile.confidence_min}, {profile.confidence_max}]")
        except (TypeError, ValueError):
            errors.append(f"confidence '{fm.get('confidence')}' is not a number")

    domain = str(fm.get("domain", "")).strip().lower()
    if domain and domain not in profile.domains:
        warnings.append(f"domain '{domain}' not in profile ({', '.join(sorted(profile.domains))})")

    for lf in profile.list_fields:
        if lf in fm and fm.get(lf) is not None and not isinstance(fm[lf], list):
            errors.append(f"field '{lf}' must be a list, got {type(fm[lf]).__name__}")

    if "type" in fm and fm.get("type") is not None and not isinstance(fm["type"], str):
        warnings.append("OKF 'type' should be a string")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def enforce(
    page_id: str,
    frontmatter: dict,
    *,
    profile: Profile = DEFAULT_PROFILE,
    mode: str = "warn",
) -> ValidationResult:
    """Apply the contract per `mode`. Returns the result; raises `ProfileViolation`
    only in strict mode with hard errors. `off` skips validation entirely."""
    if (mode or "warn").lower() == "off":
        return ValidationResult(ok=True)
    result = validate_page(frontmatter, profile=profile)
    if not result.ok and (mode or "warn").lower() == "strict":
        raise ProfileViolation(page_id, result.errors)
    return result
