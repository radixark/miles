"""Sandbox health: reports from agent code, subproc relay, bounded table, snapshot, endpoint and flush robustness."""

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.sandbox_reports import push_sandbox_reports, report_sandbox
from miles.utils import observers as observers_module
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers
from miles.utils.test_utils import sandbox_report_stubs
from miles_dashboard.collector import CollectorConfig, DashboardCollector
from miles_dashboard.dump_reader import DumpReader
from miles_dashboard.sandbox_health import (
    SNAPSHOT_FILENAME,
    SandboxHealth,
    SandboxReport,
    read_snapshot,
    sandbox_summary,
)
from miles_dashboard.server import make_app
from miles_dashboard.store import MetricsRecord, MetricStore, Stream


class _SandboxRecorder(Observer):
    def __init__(self):
        self.reports: list[dict] = []

    def sandbox_report(self, report):
        self.reports.append(report)


@pytest.fixture
def recorder(monkeypatch) -> _SandboxRecorder:
    observer = _SandboxRecorder()
    monkeypatch.setattr(
        observers_module,
        "entry_points",
        lambda group: [SimpleNamespace(name="recorder", load=lambda: lambda: observer)],
    )
    SingletonMeta._instances.pop(Observers, None)
    return observer


def test_report_is_a_noop_without_observers():
    report_sandbox("a", status="error", error="boom")


def test_report_coerces_values_and_swallows_a_bad_status(recorder: _SandboxRecorder):
    key = uuid.uuid4()
    report_sandbox(key, task=Path("t"), provider="modal", sandbox_id=7, status="error", error=OSError("x"))
    report_sandbox("a", status="exploded")
    [report] = recorder.reports
    assert (report["sandbox_key"], report["task"], report["sandbox_id"], report["status"], report["error"]) == (
        str(key),
        "t",
        "7",
        "error",
        "x",
    )


def test_subproc_reports_reach_the_callers_observers(recorder: _SandboxRecorder):
    result = asyncio.run(
        call_agent_function(
            sandbox_report_stubs.report_sandbox_health,
            mode="subproc",
            on_child_reports=push_sandbox_reports,
            sandbox_key="a",
        )
    )
    assert result == "reported"
    assert [(r["sandbox_key"], r["status"], r["provider"]) for r in recorder.reports] == [
        ("a", "healthy", "local"),
        ("a", "finished", None),
    ]


def test_a_hung_subproc_call_reports_before_it_returns_and_its_cancel_teardown(recorder: _SandboxRecorder):
    """A sandbox stuck in setup must show up while its call runs, and its teardown after a cancel must arrive."""

    async def run() -> None:
        call = asyncio.create_task(
            call_agent_function(
                sandbox_report_stubs.report_then_hang,
                mode="subproc",
                on_child_reports=push_sandbox_reports,
                sandbox_key="stuck",
            )
        )
        for _ in range(300):
            if recorder.reports:
                break
            await asyncio.sleep(0.05)
        assert [r["status"] for r in recorder.reports] == ["healthy"]
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        for _ in range(300):
            if len(recorder.reports) == 2:
                break
            await asyncio.sleep(0.05)

    asyncio.run(run())
    assert [(r["sandbox_key"], r["status"]) for r in recorder.reports] == [("stuck", "healthy"), ("stuck", "finished")]


def _report(ts: float, key: str, status: str = "healthy", **kwargs) -> SandboxReport:
    return SandboxReport(ts=ts, sandbox_key=key, status=status, **kwargs)


def test_a_row_keeps_labels_and_last_error_and_ignores_a_late_status(tmp_path: Path):
    health = SandboxHealth(tmp_path / SNAPSHOT_FILENAME)
    health.update(_report(1.0, "a", task="t", provider="modal"))
    health.update(_report(5.0, "a", "error", error="boom"))
    health.update(_report(3.0, "a", "healthy", sandbox_id="sb-1"))  # relayed late
    row = health.rows["a"]
    assert (row.status, row.last_error, row.task, row.provider, row.sandbox_id) == (
        "error",
        "boom",
        "t",
        "modal",
        "sb-1",
    )
    assert (row.first_seen_ts, row.last_exec_ts) == (1.0, 5.0)


def test_a_sandbox_torn_down_after_an_error_stays_an_error_until_it_expires(tmp_path: Path):
    health = SandboxHealth(tmp_path / SNAPSHOT_FILENAME, finished_ttl_s=60.0)
    health.update(_report(1.0, "a"))
    health.update(_report(2.0, "a", "error", error="boom"))
    health.update(_report(2.1, "a", "finished"))
    assert (health.rows["a"].status, health.rows["a"].finished) == ("error", True)
    health.write_snapshot(now=100.0)
    assert not health.rows


def test_finished_rows_expire_and_the_table_is_capped(tmp_path: Path):
    health = SandboxHealth(tmp_path / SNAPSHOT_FILENAME, finished_ttl_s=60.0, max_rows=3)
    health.update(_report(0.0, "done", "finished"))
    health.update(_report(0.0, "old"))
    health.write_snapshot(now=100.0)
    assert set(health.rows) == {"old"}, "a finished row past its TTL is evicted"
    for i in range(5):
        health.update(_report(200.0 + i, f"new-{i}"))
    assert set(health.rows) == {"new-2", "new-3", "new-4"}, "the longest-silent rows go past the cap"


def test_snapshot_round_trip_is_atomic_and_its_time_advances_without_new_reports(tmp_path: Path):
    """A static view measures silence up to the last flush, so an all-silent run still shows stale rows."""
    path = tmp_path / "dashboard" / SNAPSHOT_FILENAME
    health = SandboxHealth(path)
    health.write_snapshot(now=1.0)
    assert read_snapshot(path) is None, "no file before any sandbox reports"
    health.update(_report(1.0, "a", task="t"))
    health.write_snapshot(now=2.0)
    ts, [row] = read_snapshot(path)
    assert (ts, row["sandbox_key"], row["task"]) == (2.0, "a", "t")
    assert not path.with_name(SNAPSHOT_FILENAME + ".tmp").exists()
    health.write_snapshot(now=3.0)
    assert read_snapshot(path)[0] == 3.0


def test_summary_counts_stale_healthy_rows_and_lists_problems_first():
    rows = [
        dict(sandbox_key="ok", status="healthy", last_exec_ts=95.0),
        dict(sandbox_key="quiet", status="healthy", last_exec_ts=10.0),
        dict(sandbox_key="bad", status="error", last_exec_ts=90.0),
        dict(sandbox_key="done", status="finished", last_exec_ts=0.0),
    ]
    summary = sandbox_summary(rows, now=100.0, stale_after_s=60.0)
    assert summary["counts"] == dict(healthy=1, error=1, finished=1, stale=1)
    assert [r["sandbox_key"] for r in summary["sandboxes"]] == ["bad", "quiet", "ok", "done"]


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    MetricStore(tmp_path / "dashboard").flush()
    health = SandboxHealth(tmp_path / "dashboard" / SNAPSHOT_FILENAME)
    health.update(_report(0.0, "quiet"))
    health.update(_report(150.0, "bad", "error", error="boom"))
    health.write_snapshot(now=200.0)
    return tmp_path


def test_static_endpoint_judges_staleness_at_snapshot_time(dump_dir: Path):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    assert client.get("/api/meta").json()["capabilities"]["has_sandboxes"] is True
    summary = client.get("/api/sandboxes", params=dict(stale_after_s=100)).json()
    assert (summary["now"], summary["counts"]["stale"], summary["counts"]["error"]) == (200.0, 1, 1)
    assert client.get("/api/sandboxes", params=dict(stale_after_s=500)).json()["counts"]["healthy"] == 1
    for bad in (0, -1, "inf", "nan"):
        assert client.get("/api/sandboxes", params=dict(stale_after_s=bad)).status_code == 400


def test_follow_endpoint_uses_wall_clock_and_serves_an_empty_run(dump_dir: Path, tmp_path_factory):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir), follow=True))
    assert client.get("/api/sandboxes", params=dict(stale_after_s=1e6)).json()["counts"]["stale"] == 1
    empty = tmp_path_factory.mktemp("empty")
    client = TestClient(make_app(MetricStore(empty / "dashboard"), DumpReader(empty), follow=True))
    assert client.get("/api/meta").json()["capabilities"]["has_sandboxes"] is False
    assert client.get("/api/sandboxes").json()["sandboxes"] == []


def test_collector_writes_the_snapshot_on_flush(tmp_path: Path):
    collector = DashboardCollector(config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0))
    collector.push_sandbox_reports([_report(1.0, "a", provider="docker"), _report(2.0, "a", "error", error="x")])
    collector.flush()
    _, [row] = read_snapshot(tmp_path / SNAPSHOT_FILENAME)
    assert (row["status"], row["provider"], row["last_error"]) == ("error", "docker", "x")
    assert json.loads((tmp_path / SNAPSHOT_FILENAME).read_text())["ts"] > 0


def test_flush_drops_a_record_it_cannot_encode_and_keeps_the_rest(tmp_path: Path):
    collector = DashboardCollector(config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0))
    good = [MetricsRecord(ts=float(i), step_key=None, step=i, metrics={"m": 1.0}) for i in range(2)]
    bad = MetricsRecord(ts=0.5, step_key=None, step=9, metrics={"m": np.float32(1.0)})
    for record in (good[0], bad, good[1]):
        collector.push_metrics(record)
    collector.flush()
    assert MetricStore.load(tmp_path).records[Stream.METRICS] == good


def test_flush_loop_survives_a_failing_flush(tmp_path: Path, monkeypatch):
    collector = DashboardCollector(
        config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0, flush_interval_seconds=0.01)
    )
    calls = []

    def failing_flush():
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(collector, "flush", failing_flush)
    monkeypatch.setattr(collector, "_reconcile_samplers", lambda: None)
    collector.start()
    for _ in range(200):
        if len(calls) >= 2:
            break
        asyncio.run(asyncio.sleep(0.01))
    collector._stop_event.set()
    collector._flush_thread.join(timeout=1)
    assert len(calls) >= 2
