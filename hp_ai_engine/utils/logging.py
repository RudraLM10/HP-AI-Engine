"""
Structured logging for HP AI Engine.

Provides JSON-formatted log output with contextual fields (station_id, cluster_id)
for observability across the distributed three-tier architecture.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include any extra context fields attached to the record
        for key in ("station_id", "cluster_id", "tier", "component"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects context fields into log records.

    Usage:
        logger = get_logger("engine", station_id="ST_001", cluster_id="Mumbai_West")
        logger.info("Forecast generated", extra={"horizon": "6h"})
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    stream: Any = None,
) -> None:
    """
    Configure root logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON formatter. If False, use standard format.
        stream: Output stream. Defaults to sys.stdout.
    """
    root_logger = logging.getLogger("hp_ai_engine")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stdout)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))

    root_logger.addHandler(handler)


def get_logger(
    name: str,
    **context: Any,
) -> ContextAdapter:
    """
    Get a context-aware logger.

    Args:
        name: Logger name (will be prefixed with 'hp_ai_engine.').
        **context: Contextual fields to attach to every log record.
            Common fields: station_id, cluster_id, tier, component.

    Returns:
        ContextAdapter wrapping the named logger.

    Example:
        logger = get_logger("gcn_encoder", component="prediction_engine")
        logger.info("Forward pass completed", extra={"num_nodes": 50})
    """
    logger = logging.getLogger(f"hp_ai_engine.{name}")
    return ContextAdapter(logger, extra=context)
