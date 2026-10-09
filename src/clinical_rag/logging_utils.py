"""Structured JSON logging.

Log records never include raw questions or answers, only identifiers, counts and timings,
so that accidental PHI in user input does not end up in log storage.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects, including ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit JSON to stderr (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, "_crag_json", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._crag_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
