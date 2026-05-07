import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from miles.utils.event_logger.logger import EventLogger, get_event_logger, set_event_logger
from miles.utils.event_logger.models import WitnessAllocateIdEvent
from miles.utils.process_identity import MainProcessIdentity, TrainProcessIdentity

_TEST_SOURCE = MainProcessIdentity()


def _make_logger(log_dir: Path, file_name: str = "events.jsonl") -> EventLogger:
    return EventLogger(log_dir=log_dir, file_name=file_name, source=_TEST_SOURCE)


_EVENT_CLS = WitnessAllocateIdEvent
_EVENT_PARTIAL: dict = dict(rollout_id=0, attempt=0, witness_id_to_sample_index={10: 0, 11: 1, 12: 2})


class TestEventLoggerWritesJsonl:
    def test_writes_multiple_events(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, file_name="test.jsonl")

        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.log(WitnessAllocateIdEvent, dict(rollout_id=1, attempt=0, witness_id_to_sample_index={0: 0}))
        logger.close()

        lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "type" in parsed


class TestEventLoggerAutoFillsMetadata:
    def test_timestamp_is_utc_and_recent(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)

        before = datetime.now(timezone.utc)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        after = datetime.now(timezone.utc)
        logger.close()

        line = (tmp_path / "events.jsonl").read_text().strip()
        parsed = json.loads(line)
        ts = datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))
        assert before <= ts <= after

    def test_source_auto_filled(self, tmp_path: Path) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=3)
        logger = EventLogger(log_dir=tmp_path, source=source)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["source"]["component"] == "actor"
        assert parsed["source"]["cell_index"] == 2
        assert parsed["source"]["rank_within_cell"] == 3


class TestEventLoggerThreadSafety:
    def test_concurrent_writes_no_data_loss(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        num_threads = 8
        events_per_thread = 50

        def writer() -> None:
            for _ in range(events_per_thread):
                logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        threads = [threading.Thread(target=writer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == num_threads * events_per_thread

        for line in lines:
            json.loads(line)


class TestSetGetEventLogger:
    def test_set_then_get(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        set_event_logger(logger)
        assert get_event_logger() is logger
        logger.close()
        set_event_logger(None)

    def test_replace_logger(self, tmp_path: Path) -> None:
        logger1 = _make_logger(tmp_path, file_name="a.jsonl")
        logger2 = _make_logger(tmp_path, file_name="b.jsonl")
        set_event_logger(logger1)
        set_event_logger(logger2)
        assert get_event_logger() is logger2
        logger1.close()
        logger2.close()
        set_event_logger(None)


class TestGetEventLoggerRaisesWhenNotSet:
    def test_raises_runtime_error(self) -> None:
        import miles.utils.event_logger.logger as mod

        original = mod._event_logger
        mod._event_logger = None
        try:
            with pytest.raises(RuntimeError, match="EventLogger not initialized"):
                get_event_logger()
        finally:
            mod._event_logger = original


class TestEventLoggerFlushOnEachWrite:
    def test_readable_before_close(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        content = (tmp_path / "events.jsonl").read_text()
        assert len(content.strip()) > 0
        logger.close()


class TestEventLoggerCreatesDirectory:
    def test_creates_nested_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        logger = _make_logger(nested)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()
        assert (nested / "events.jsonl").exists()


class TestEventLoggerClose:
    def test_file_closed_after_close(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()
        assert logger._file.closed


class TestReadEvents:
    def test_malformed_line_skipped_with_warning(self, tmp_path: Path) -> None:
        from miles.utils.event_logger.logger import read_events

        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        with open(tmp_path / "events.jsonl", "a") as f:
            f.write("this is not valid json\n")

        events = read_events(tmp_path)
        assert len(events) == 1

    def test_reads_multiple_jsonl_files(self, tmp_path: Path) -> None:
        from miles.utils.event_logger.logger import read_events

        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_TEST_SOURCE)
        logger_a.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_TEST_SOURCE)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.close()

        events = read_events(tmp_path)
        assert len(events) == 3
