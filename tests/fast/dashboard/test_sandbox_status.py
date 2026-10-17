"""Sandbox lifecycle events: store round trip, aggregation, endpoint and report hook."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miles.dashboard import backend, hooks
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.sandbox_status import sandbox_status
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, SandboxEvent, SandboxStage, Stream


def _event(ts: float, trial_id: str, stage: SandboxStage, exception_type: str | None = None) -> SandboxEvent:
    return SandboxEvent(ts=ts, trial_id=trial_id, task="t", stage=stage.value, exception_type=exception_type)


EVENTS = [
    _event(0.0, "a", SandboxStage.CREATED),
    _event(5.0, "a", SandboxStage.AGENT_STARTED),
    _event(1.0, "b", SandboxStage.CREATED),
    _event(70.0, "b", SandboxStage.FAILED, "TimeoutError"),
    _event(2.0, "c", SandboxStage.CREATED),
    _event(130.0, "c", SandboxStage.FAILED, "TimeoutError"),
    _event(3.0, "d", SandboxStage.CREATED),
    _event(75.0, "d", SandboxStage.FAILED),
    _event(4.0, "e", SandboxStage.TORN_DOWN),
    _event(200.0, "f", SandboxStage.SETUP_DONE),
]


def test_stage_counts_use_each_trials_latest_event_regardless_of_arrival_order():
    status = sandbox_status(list(reversed(EVENTS)), now=200.0, stale_after_s=600.0, bucket_s=60.0)
    assert status["num_trials"] == 6
    counts = status["stage_counts"]
    assert (counts["agent_started"], counts["failed"], counts["torn_down"], counts["setup_done"]) == (1, 3, 1, 1)
    assert counts["created"] == 0, "a superseded stage must not be counted"


def test_failures_are_bucketed_per_exception_type_with_a_missing_type_as_unknown():
    status = sandbox_status(EVENTS, now=200.0, stale_after_s=600.0, bucket_s=60.0)
    assert status["failure_counts"] == {"TimeoutError": 2, "unknown": 1}
    series = {s["label"]: s for s in status["failure_series"]}
    assert series["TimeoutError"]["ts"] == [70.0, 130.0]
    assert series["TimeoutError"]["value"] == [1, 1]
    assert series["unknown"]["value"] == [1, 0]


def test_only_silent_non_terminal_trials_are_stale():
    status = sandbox_status(EVENTS, now=200.0, stale_after_s=60.0, bucket_s=60.0)
    assert [row["trial_id"] for row in status["stale"]] == ["a"]
    assert status["stale"][0]["silent_s"] == 195.0


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    store = MetricStore(tmp_path / "dashboard")
    for event in EVENTS:
        store.append(event)
    store.flush()
    return tmp_path


def test_static_endpoint_judges_staleness_at_the_last_recorded_event(dump_dir: Path):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    assert client.get("/api/meta").json()["capabilities"]["has_sandbox_events"] is True
    status = client.get("/api/sandboxes", params=dict(stale_after_s=60)).json()
    assert status["now"] == 200.0
    assert [row["trial_id"] for row in status["stale"]] == ["a"]
    assert client.get("/api/sandboxes", params=dict(bucket_s=0)).status_code == 400


def test_follow_endpoint_judges_staleness_at_wall_clock(dump_dir: Path):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir), follow=True))
    stale = client.get("/api/sandboxes").json()["stale"]
    assert {row["trial_id"] for row in stale} == {"a", "f"}


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)


class _FakeHandle:
    def __init__(self):
        self.push_sandbox_events = _FakeRemoteMethod()


def test_report_is_a_noop_without_a_collector(monkeypatch):
    monkeypatch.setattr(backend, "_handle", None)
    hooks.report_sandbox_event("a", SandboxStage.CREATED)


def test_report_pushes_to_the_collector_and_swallows_a_bad_stage(monkeypatch):
    handle = _FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.report_sandbox_event("a", SandboxStage.FAILED, task="t", exception_type="OSError", provider="modal")
    hooks.report_sandbox_event("a", "not-a-stage")
    [([event],)] = handle.push_sandbox_events.calls
    assert (event.trial_id, event.stage, event.exception_type, event.provider) == ("a", "failed", "OSError", "modal")


def test_collector_writes_pushed_events_to_disk(tmp_path: Path):
    collector = DashboardCollector(config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0))
    collector.push_sandbox_events(EVENTS[:3])
    collector.flush()
    assert MetricStore.load(tmp_path).records[Stream.SANDBOX_EVENTS] == EVENTS[:3]
