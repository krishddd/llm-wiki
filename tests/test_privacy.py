"""Privacy / secret redaction tests — secrets must be stripped before ingest."""
from __future__ import annotations

from llm_wiki.privacy import redact_text


def test_redacts_openai_style_key() -> None:
    res = redact_text("Use sk-abcd1234EFGH5678ijklmnop as the key.")
    assert "sk-abcd1234" not in res.text
    assert "[REDACTED:api_key]" in res.text
    assert res.counts.get("api_key") == 1


def test_redacts_github_pat() -> None:
    res = redact_text("token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 here")
    assert "ghp_" not in res.text
    assert res.counts.get("api_key") == 1


def test_redacts_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    res = redact_text(f"Authorization: Bearer {jwt}")
    assert jwt not in res.text
    assert res.counts.get("jwt") == 1


def test_redacts_private_key_block() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    res = redact_text(f"key:\n{pem}\ndone")
    assert "BEGIN RSA PRIVATE KEY" not in res.text
    assert res.counts.get("private_key") == 1


def test_redacts_plaintext_password_keeps_field_name() -> None:
    res = redact_text("password: hunter2secret")
    assert "hunter2secret" not in res.text
    assert "password" in res.text.lower()          # field name preserved for context
    assert res.counts.get("password") == 1


def test_emails_preserved_by_default() -> None:
    res = redact_text("Contact author jane.doe@example.com for details.")
    assert "jane.doe@example.com" in res.text       # public author emails kept
    assert not res.redacted


def test_emails_redacted_when_opted_in() -> None:
    res = redact_text("Contact jane.doe@example.com", redact_emails=True)
    assert "jane.doe@example.com" not in res.text
    assert res.counts.get("email") == 1


def test_clean_text_untouched() -> None:
    text = "The free energy principle minimizes variational free energy."
    res = redact_text(text)
    assert res.text == text
    assert not res.redacted
    assert res.total == 0


def test_multiple_secrets_counted() -> None:
    res = redact_text("sk-abcd1234EFGH5678ijklmnop and ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    assert res.counts.get("api_key") == 2
    assert res.total == 2
