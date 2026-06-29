"""Disk-based JSONL logger for session records.

Each session gets its own file: ``<log_dir>/<session_id>.jsonl``.
Writes are synchronous — each ``log_record`` call serializes and flushes
immediately.
"""

import json
import logging
from pathlib import Path
from typing import IO

from miles.rollout.session.session_types import SessionRecord

logger = logging.getLogger(__name__)


class RecordLogger:
    """Writes ``SessionRecord`` objects as one-JSON-per-line to per-session files."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, IO[str]] = {}
        logger.info("[record-logger] Logging session records to %s", self.log_dir)

    def log_record(self, session_id: str, record: SessionRecord) -> None:
        handle = self._handles.get(session_id)
        if handle is None:
            path = self.log_dir / f"{session_id}.jsonl"
            handle = open(path, "a", encoding="utf-8")
            self._handles[session_id] = handle
        handle.write(json.dumps(record.model_dump(), default=str) + "\n")
        handle.flush()

    def close_session(self, session_id: str) -> None:
        handle = self._handles.pop(session_id, None)
        if handle is not None:
            handle.close()

    def close_all(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
