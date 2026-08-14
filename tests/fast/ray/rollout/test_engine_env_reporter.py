from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import engine_env_reporter as engine_env_reporter_module
from miles.ray.rollout.engine_env_reporter import RETRY_INTERVAL_SECONDS, EngineEnvReporter
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_SAMPLE_SERVER_INFO: dict[str, Any] = {
    "model_path": "/models/qwen",
    "api_key": "hunter2",
    "admin_api_key": "hunter3",
    "ssl_keyfile_password": "hunter4",
    "version": "0.5.0",
    "internal_states": [
        {
            "last_gen_throughput": 1.0,
            "api_key": "hunter2",
            "admin_api_key": "hunter3",
            "env_vars": {"HF_TOKEN": "hunter5", "CUDA_VISIBLE_DEVICES": "0"},
        },
        {"last_gen_throughput": 2.0, "api_key": "hunter2", "env_vars": {"RANK": "1"}},
    ],
}


class _FakeApiClient:
    def __init__(self, *, server_info: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.call_count = 0
        self.error = error
        self._server_info = _SAMPLE_SERVER_INFO if server_info is None else server_info

    async def get_server_info(self) -> dict[str, Any]:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self._server_info


@pytest.fixture()
def event_log_dir(tmp_path: Path) -> Path:
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="inference_controller")))
    yield tmp_path
    set_event_logger(None)


@pytest.fixture()
def fake_clock(monkeypatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(engine_env_reporter_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def _report(reporter: EngineEnvReporter, api_client: _FakeApiClient, *, cell_id: str = "a") -> None:
    await reporter.report_if_due(cell_id=cell_id, server_url=f"http://{cell_id}:30000", api_client=api_client)


def _events(log_dir: Path) -> list[EngineEnvReportEvent]:
    return read_events(log_dir)


class TestReportIfDue:
    async def test_records_one_event_for_the_cell(self, event_log_dir: Path) -> None:
        """An engine's environment is only knowable once it answers, so the reading is the record."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        events = _events(event_log_dir)
        assert [event.cell_id for event in events] == ["a"]
        assert events[0].server_url == "http://a:30000"
        assert events[0].source == SimpleProcessIdentity(component="inference_controller")

    async def test_reads_only_once_within_the_interval(self, event_log_dir: Path, fake_clock) -> None:
        """The sweep runs every few seconds, and re-dumping a whole environment each time would flood the log."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 3599.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_reads_again_after_the_interval(self, event_log_dir: Path, fake_clock) -> None:
        """An engine's environment can be changed under it, so one reading at startup is not enough."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 3600.0
        await _report(reporter, api_client)

        assert api_client.call_count == 2

    async def test_a_non_positive_interval_reads_a_cell_once(self, event_log_dir: Path, fake_clock) -> None:
        """--env-report-interval-seconds of zero means 'only at startup', for engines as much as for processes."""
        reporter, api_client = EngineEnvReporter(interval_seconds=0.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 100000.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_restarted_cell_is_read_again_immediately(self, event_log_dir: Path, fake_clock) -> None:
        """Healing builds a new cell, whose reporter is new too, which is the case this audit exists for."""
        api_client = _FakeApiClient()

        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)
        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert api_client.call_count == 2


class TestRedaction:
    async def test_the_recorded_server_info_carries_no_secret(self, event_log_dir: Path) -> None:
        """/server_info returns the engine's api keys and its whole sglang environment, verbatim."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        event = _events(event_log_dir)[0]
        server_info = event.server_info
        assert "hunter" not in event.model_dump_json()
        assert server_info["api_key"].startswith("redacted-sha256:")
        assert server_info["admin_api_key"].startswith("redacted-sha256:")
        assert server_info["ssl_keyfile_password"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["env_vars"]["HF_TOKEN"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["api_key"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["admin_api_key"].startswith("redacted-sha256:")

    async def test_the_recorded_server_info_keeps_everything_that_is_not_a_secret(self, event_log_dir: Path) -> None:
        """A redaction that also drops the model path and the engine's rank records nothing worth auditing."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        server_info = _events(event_log_dir)[0].server_info
        assert server_info["model_path"] == "/models/qwen"
        assert server_info["version"] == "0.5.0"
        assert server_info["internal_states"][0]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert server_info["internal_states"][1]["env_vars"]["RANK"] == "1"
        assert server_info["internal_states"][1]["last_gen_throughput"] == 2.0


class TestDegradation:
    async def test_an_engine_without_the_env_var_gate_is_still_recorded(self, event_log_dir: Path) -> None:
        """An older engine has no env_vars key at all, and its server args are still worth recording."""
        server_info = {
            "model_path": "/models/qwen",
            "api_key": "hunter2",
            "internal_states": [{"waiting_queue": 0, "api_key": "hunter2"}],
        }

        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(server_info=server_info))

        recorded = _events(event_log_dir)[0].server_info
        assert recorded["model_path"] == "/models/qwen"
        assert recorded["api_key"].startswith("redacted-sha256:")
        assert recorded["internal_states"][0]["waiting_queue"] == 0

    async def test_an_engine_without_the_env_var_gate_still_has_its_nested_keys_hidden(
        self, event_log_dir: Path
    ) -> None:
        """Every internal state repeats the whole ServerArgs, credentials included, gate or no gate."""
        server_info = {"internal_states": [{"waiting_queue": 0, "api_key": "hunter2"}]}

        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(server_info=server_info))

        event = _events(event_log_dir)[0]
        assert "hunter2" not in event.model_dump_json()
        assert event.server_info["internal_states"][0]["api_key"].startswith("redacted-sha256:")

    async def test_an_unreachable_engine_only_warns(self, event_log_dir: Path, caplog) -> None:
        """This audit rides on the controller's tick, which must not die because one engine timed out."""
        api_client = _FakeApiClient(error=TimeoutError("no answer"))

        with caplog.at_level(logging.WARNING, logger="miles.ray.rollout.engine_env_reporter"):
            await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert "Failed to record the engine env of cell a" in caplog.text
        assert caplog.records[0].exc_info is not None
        assert _events(event_log_dir) == []

    async def test_a_failed_reading_is_not_retried_on_every_tick(self, event_log_dir: Path, fake_clock) -> None:
        """The sweep runs every few seconds, and a wedged engine would warn on each of them."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(error=RuntimeError("boom"))

        await _report(reporter, api_client)
        fake_clock[0] += 5.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_failed_reading_is_retried_long_before_the_next_interval(
        self, event_log_dir: Path, fake_clock
    ) -> None:
        """A cell that answers nothing once would otherwise have no environment recorded for a whole hour."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(error=RuntimeError("boom"))

        await _report(reporter, api_client)
        fake_clock[0] += RETRY_INTERVAL_SECONDS
        api_client.error = None
        await _report(reporter, api_client)

        assert api_client.call_count == 2
        assert [event.cell_id for event in _events(event_log_dir)] == ["a"]

    async def test_a_recorded_cell_is_not_retried_at_the_failure_cadence(
        self, event_log_dir: Path, fake_clock
    ) -> None:
        """The short retry exists for failures only; a cell that answered waits out the whole interval."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += RETRY_INTERVAL_SECONDS
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_run_without_an_event_logger_does_not_raise(self, fake_clock) -> None:
        """Not every run has an event dir, and the reporter must not raise when there is none."""
        set_event_logger(None)
        api_client = _FakeApiClient()

        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert api_client.call_count == 1
