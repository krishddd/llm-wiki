"""Privacy / secret redaction — applied to raw source text BEFORE ingest.

Implements the policy documented in CLAUDE.md ("Privacy filtering"): sources may
contain PII / credentials, so strip them before the text reaches the summariser,
the claim extractor, the graph, the embeddings, or the on-disk wiki page.

Each redaction replaces the secret with a typed placeholder (e.g. `[REDACTED:api_key]`)
so the surrounding text stays coherent for the LLM while the secret never persists.
Callers should audit-log a `PRIVACY_REDACT` event with the per-category counts.

Deliberately conservative: every pattern targets a high-precision secret shape.
We would rather miss an exotic token than corrupt ordinary prose — false positives
silently degrade ingest quality, which is hard to debug later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def redacted(self) -> bool:
        return self.total > 0


# ── Patterns ────────────────────────────────────────────────────────────────
# Order matters: more specific provider tokens run before the generic catch-alls
# so a Slack/GitHub/OpenAI key is labelled precisely rather than as "secret".
#
# Each entry is (category, compiled-pattern). The whole match is replaced.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Provider API keys / tokens — distinctive prefixes.
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),                  # OpenAI / Anthropic style
    ("api_key", re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b")),  # GitHub PAT
    ("api_key", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),           # Slack
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),                       # AWS access key id
    ("api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),                  # Google API key
    ("api_key", re.compile(r"\b(?:glpat|gldt)-[A-Za-z0-9_-]{20,}\b")),      # GitLab PAT
    # JSON Web Tokens — three base64url segments.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    # Private key PEM blocks.
    ("private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
        re.DOTALL,
    )),
    # Plaintext passwords assigned in `password: hunter2` / `pwd=hunter2` form.
    ("password", re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\b\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?",
    )),
    # Email addresses (PII). The policy keeps public author emails; we cannot know
    # publicity at this layer, so redaction is gated by `redact_emails` (default off).
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

_PLACEHOLDER = "[REDACTED:{cat}]"


def redact_text(text: str, *, redact_emails: bool = False) -> RedactionResult:
    """Strip secrets/PII from `text`, returning the cleaned text + per-category counts.

    `redact_emails` is opt-in: paper-author / public emails are legitimate wiki
    content, so we only strip them when the caller explicitly asks.
    """
    if not text:
        return RedactionResult(text=text or "", counts={})

    counts: dict[str, int] = {}
    out = text
    for category, pattern in _PATTERNS:
        if category == "email" and not redact_emails:
            continue

        def _sub(_m: re.Match[str], _cat: str = category) -> str:
            counts[_cat] = counts.get(_cat, 0) + 1
            return _PLACEHOLDER.format(cat=_cat)

        # The "password" pattern matches `key: value` — replace only the value group
        # so the field name is preserved for context.
        if category == "password":
            def _sub_pwd(m: re.Match[str]) -> str:
                counts["password"] = counts.get("password", 0) + 1
                return m.group(0).replace(m.group(1), _PLACEHOLDER.format(cat="password"))
            out = pattern.sub(_sub_pwd, out)
        else:
            out = pattern.sub(_sub, out)

    return RedactionResult(text=out, counts=counts)
