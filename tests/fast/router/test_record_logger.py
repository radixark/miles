"""Tests for RecordLogger."""

from __future__ import annotations

import json
from pathlib import Path


from miles.rollout.session.record_logger import RecordLogger
from miles.rollout.session.session_types import SessionRecord


def _make_record(index: int, session_tag: str = "default") -> SessionRecord:
    return SessionRecord(
        timestamp=1000.0 + index,
        method="POST",
        path=f"/api/{session_tag}/{index}",
        request={"index": index, "session_tag": session_tag},
        response={"ok": True, "index": index},
        status_code=200,
    )


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class TestRecordLogger:
    def test_write_and_readback(self, tmp_path: Path):
        rl = RecordLogger(str(tmp_path))
        for i in range(5):
            rl.log_record("s1", _make_record(i, "s1"))
        rl.close_all()

        records = _read_jsonl(tmp_path / "s1.jsonl")
        assert len(records) == 5
        assert [r["timestamp"] for r in records] == [1000.0 + i for i in range(5)]

    def test_multiple_sessions(self, tmp_path: Path):
        rl = RecordLogger(str(tmp_path))
        for i in range(3):
            rl.log_record("a", _make_record(i, "a"))
            rl.log_record("b", _make_record(i, "b"))
        rl.close_all()

        assert len(_read_jsonl(tmp_path / "a.jsonl")) == 3
        assert len(_read_jsonl(tmp_path / "b.jsonl")) == 3

    def test_close_session_then_reopen(self, tmp_path: Path):
        rl = RecordLogger(str(tmp_path))
        rl.log_record("s", _make_record(0))
        rl.close_session("s")
        rl.log_record("s", _make_record(1))
        rl.close_all()

        records = _read_jsonl(tmp_path / "s.jsonl")
        assert len(records) == 2

    def test_close_nonexistent_session_is_noop(self, tmp_path: Path):
        rl = RecordLogger(str(tmp_path))
        rl.close_session("nonexistent")  # should not raise
        rl.close_all()

    def test_close_all_clears_handles(self, tmp_path: Path):
        rl = RecordLogger(str(tmp_path))
        rl.log_record("s", _make_record(0))
        rl.close_all()
        assert len(rl._handles) == 0
