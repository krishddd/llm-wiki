"""SOC-style structured JSON logging with correlation-id injection via contextvars."""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

_event_counter = 0


def new_correlation_id() -> str:
    return "COR-" + uuid.uuid4().hex[:12].upper()


def new_event_id() -> str:
    global _event_counter
    _event_counter += 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"EVT-{ts}-{_event_counter:04d}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "event_id": new_event_id(),
            "correlation_id": correlation_id_ctx.get(),
            "severity": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for k in ("event_type", "actor", "resource", "outcome", "duration_ms"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        meta = getattr(record, "metadata", None)
        if meta:
            payload["metadata"] = meta
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_AUDIT_EVENTS = {
    "WIKI_WRITE",
    "WIKI_WRITE_STAGED",
    "WIKI_REVIEW_ACCEPT",
    "WIKI_REVIEW_REJECT",
    "CONTRADICTION_DETECTED",
    "CONFIDENCE_LOW",
    "STALE_PAGE_DETECTED",
    "ORPHAN_PAGE_DETECTED",
}


class AuditFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "event_type", None) in _AUDIT_EVENTS


_configured = False


def setup_logging(logs_dir: Path) -> None:
    global _configured
    if _configured:
        return
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = JsonFormatter()

    app_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    app_handler.setFormatter(formatter)
    root.addHandler(app_handler)

    audit_handler = logging.FileHandler(logs_dir / "audit.log", encoding="utf-8")
    audit_handler.setFormatter(formatter)
    audit_handler.addFilter(AuditFilter())
    root.addHandler(audit_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    _configured = True


def audit(logger: logging.Logger, event_type: str, resource: str, outcome: str = "SUCCESS", **metadata) -> None:
    """Emit an audit-grade log entry. Always written to audit.log."""
    logger.info(
        metadata.pop("message", event_type),
        extra={"event_type": event_type, "resource": resource, "outcome": outcome, "metadata": metadata},
    )


def timed_ms(start_perf: float) -> float:
    return round((time.perf_counter() - start_perf) * 1000, 2)


def rand_token() -> str:
    return secrets.token_hex(4)
