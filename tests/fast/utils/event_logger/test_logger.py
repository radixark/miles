import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import miles.utils.audit_utils.event_logger.logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import (
    EventLogger,
    event_logger_context,
    get_event_logger,
    set_event_logger,
)
from miles.utils.audit_utils.event_logger.models import MetricEvent, WitnessAllocateIdEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity, TrainProcessIdentity

_TEST_SOURCE = MainProcessIdentity()


def _make_logger(log_dir: Path, file_name: str = "events.jsonl") -> EventLogger:
    return EventLogger(log_dir=log_dir, file_name=file_name, source=_TEST_SOURCE)


_EVENT_CLS = WitnessAllocateIdEvent
_EVENT_PARTIAL: dict = dict(
    rollout_id=0, attempt=0, witness_id_to_sample_index={10: 0, 11: 1, 12: 2}, counter_after=13
)


class TestEventLoggerWritesJsonl:
    def test_writes_multiple_events(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path, file_name="test.jsonl")

        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.log(
            WitnessAllocateIdEvent, dict(rollout_id=1, attempt=0, witness_id_to_sample_index={0: 0}, counter_after=1)
        )
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
        original = event_logger_module._event_logger
        event_logger_module._event_logger = None
        try:
            with pytest.raises(RuntimeError, match="EventLogger not initialized"):
                get_event_logger()
        finally:
            event_logger_module._event_logger = original


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


class TestEventLoggerFilePerWrite:
    def test_log_after_events_file_removed_writes_to_fresh_file(self, tmp_path: Path) -> None:
        """The file is opened per write, so a checkpoint restore can swap it between events."""
        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        path = tmp_path / "events.jsonl"
        path.unlink()

        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        assert len(path.read_text().strip().split("\n")) == 1


class TestReadEvents:
    def test_malformed_line_skipped_with_warning(self, tmp_path: Path) -> None:
        from miles.utils.audit_utils.event_logger.logger import read_events

        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        with open(tmp_path / "events.jsonl", "a") as f:
            f.write("this is not valid json\n")

        events = read_events(tmp_path)
        assert len(events) == 1

    def test_reads_multiple_jsonl_files(self, tmp_path: Path) -> None:
        from miles.utils.audit_utils.event_logger.logger import read_events

        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_TEST_SOURCE)
        logger_a.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_TEST_SOURCE)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.close()

        events = read_events(tmp_path)
        assert len(events) == 3


class TestWithContext:
    def test_injects_context_fields_into_logged_event(self, tmp_path: Path) -> None:
        """Fields from with_context are merged into events logged inside the scope."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 5, "attempt": 7}):
            logger.log(MetricEvent, dict(metrics={"loss": 1.0}))
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] == 5
        assert parsed["attempt"] == 7
        assert parsed["metrics"] == {"loss": 1.0}

    def test_context_not_applied_outside_scope(self, tmp_path: Path) -> None:
        """Events logged after the context scope exits do not carry context fields."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 5}):
            pass
        logger.log(MetricEvent, dict(metrics={}))
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] is None

    def test_nesting_overrides_then_restores(self, tmp_path: Path) -> None:
        """A nested with_context overrides outer fields, then restores them on exit."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 1, "attempt": 1}):
            with logger.with_context({"attempt": 2}):
                logger.log(MetricEvent, dict(metrics={"k": "inner"}))
            logger.log(MetricEvent, dict(metrics={"k": "outer"}))
        logger.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        inner = json.loads(lines[0])
        outer = json.loads(lines[1])
        assert (inner["rollout_id"], inner["attempt"]) == (1, 2)
        assert (outer["rollout_id"], outer["attempt"]) == (1, 1)

    def test_context_var_empty_after_exit(self, tmp_path: Path) -> None:
        """After all with_context scopes exit the underlying contextvar resolves to empty."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 1}):
            with logger.with_context({"attempt": 2}):
                pass
        assert logger._context_var.get({}) == {}


class TestEventLoggerContextDecorator:
    def test_uninitialized_runs_without_context(self) -> None:
        """When the logger is uninitialized the wrapper invokes the method directly."""
        set_event_logger(None)
        calls: list[tuple] = []

        @event_logger_context(lambda obj, x: {"rollout_id": x})
        def method(obj: object, x: int) -> int:
            calls.append((obj, x))
            return x * 2

        try:
            assert method(object(), 3) == 6
            assert len(calls) == 1
        finally:
            set_event_logger(None)

    def test_ctx_fn_not_called_when_uninitialized(self) -> None:
        """ctx_fn is skipped entirely when the event logger is not initialized."""
        set_event_logger(None)
        ctx_calls: list[int] = []

        def ctx_fn(obj: object, x: int) -> dict:
            ctx_calls.append(x)
            return {"rollout_id": x}

        @event_logger_context(ctx_fn)
        def method(obj: object, x: int) -> int:
            return x

        try:
            assert method(object(), 9) == 9
            assert ctx_calls == []
        finally:
            set_event_logger(None)

    def test_initialized_injects_fields_from_method_args(self, tmp_path: Path) -> None:
        """When initialized, ctx_fn output is injected into events logged by the method."""
        logger = _make_logger(tmp_path)
        set_event_logger(logger)

        class Worker:
            @event_logger_context(lambda self, rollout_id: {"rollout_id": rollout_id})
            def run(self, rollout_id: int) -> None:
                get_event_logger().log(MetricEvent, dict(metrics={"ok": True}))

        try:
            Worker().run(42)
            logger.close()
        finally:
            set_event_logger(None)

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] == 42

    def test_ctx_fn_receives_method_args(self, tmp_path: Path) -> None:
        """ctx_fn is called with exactly the same positional/keyword args as the method."""
        logger = _make_logger(tmp_path)
        set_event_logger(logger)
        seen: list[tuple] = []

        def ctx_fn(obj: object, rollout_id: int, *, attempt: int) -> dict:
            seen.append((rollout_id, attempt))
            return {"rollout_id": rollout_id, "attempt": attempt}

        @event_logger_context(ctx_fn)
        def method(obj: object, rollout_id: int, *, attempt: int) -> None:
            get_event_logger().log(MetricEvent, dict(metrics={}))

        try:
            method(object(), 3, attempt=8)
            logger.close()
        finally:
            set_event_logger(None)

        assert seen == [(3, 8)]
        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert (parsed["rollout_id"], parsed["attempt"]) == (3, 8)
