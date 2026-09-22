"""
Structured JSON logging for the mRNA design pipeline.

Every agent action, tool call, edit proposal, and score update is logged as a
JSON event to `logs/<run_id>/events.jsonl`. This makes post-hoc analysis and
LLM prompt debugging straightforward.

Usage::

    from mrna_design.logging_utils import get_logger
    log = get_logger("structure_agent")
    log.event("fold_called", sequence_length=1200, mfe=-45.3)
    log.score("objectives_computed", cai=0.82, gc=0.55)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra fields attached via log.event() / log.score()
        for key, val in record.__dict__.items():
            if key.startswith("_extra_"):
                base[key[7:]] = val
        return json.dumps(base, default=str)


class PipelineLogger:
    """Thin wrapper around stdlib Logger that adds `.event()` and `.score()`."""

    def __init__(self, name: str, log_dir: Path | None = None) -> None:
        self._logger = logging.getLogger(name)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "events.jsonl")
            fh.setFormatter(_JsonFormatter())
            self._logger.addHandler(fh)
        # Also write to stderr as plain JSON lines (easy to pipe / grep)
        if not self._logger.handlers:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(_JsonFormatter())
            self._logger.addHandler(sh)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

    def _log(self, level: int, event_type: str, **kwargs: Any) -> None:
        extra = {f"_extra_{k}": v for k, v in kwargs.items()}
        extra["_extra_event"] = event_type
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            fn="",
            lno=0,
            msg=event_type,
            args=(),
            exc_info=None,
            extra=extra,
        )
        self._logger.handle(record)

    def event(self, event_type: str, **kwargs: Any) -> None:
        """Log a pipeline event (tool call, edit proposal, etc.)."""
        self._log(logging.INFO, event_type, **kwargs)

    def score(self, event_type: str, **kwargs: Any) -> None:
        """Log numerical scores / metrics."""
        self._log(logging.INFO, event_type, **kwargs)

    def warn(self, event_type: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event_type, **kwargs)

    def error(self, event_type: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event_type, **kwargs)

    def debug(self, event_type: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event_type, **kwargs)


# Module-level registry so every component gets a consistent logger
_registry: dict[str, PipelineLogger] = {}
_global_log_dir: Path | None = None


def configure(log_dir: Path) -> None:
    """Call once per run to set the output directory for all future loggers."""
    global _global_log_dir
    _global_log_dir = log_dir
    log_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> PipelineLogger:
    if name not in _registry:
        _registry[name] = PipelineLogger(name, _global_log_dir)
    return _registry[name]
