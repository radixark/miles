"""The dashboard observer hands each event to the collector actor as the record the store reads."""

import time
from importlib.metadata import EntryPoint
from types import SimpleNamespace

import numpy as np
import pytest

from miles.dashboard import observer as dashboard_observer
from miles.dashboard import sandbox_health
from miles.dashboard.observer import (
    BATCH_MAX_EVENTS,
    END_PROCESS_TIMEOUT_SECONDS,
    SHUTDOWN_TIMEOUT_SECONDS,
    DashboardObserver,
)
from miles.dashboard.store import EngineInfo, MetricStore, PhaseEvent, Role, Stream, TrajectoryEvent
from miles.utils import observers
from miles.utils.misc import SingletonMeta

TRAIN_RANK = dict(role="train", node="10.0.0.3", rank=7, gpus=[3])
ROLLOUT_MANAGER = dict(role="rollout_manager", node="10.0.0.5", rank=-1, gpus=[])


class FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeCollector:
    def __init__(self):
        self.push_phases = FakeRemoteMethod()
        self.push_trajectories = FakeRemoteMethod()
        self.push_metrics = FakeRemoteMethod()
        self.update_topology = FakeRemoteMethod()
        self.push_data_buffer = FakeRemoteMethod()
        self.push_sandbox_reports = FakeRemoteMethod()
        self.ping = FakeRemoteMethod()
        self.shutdown = FakeRemoteMethod()


@pytest.fixture
def collector(monkeypatch) -> FakeCollector:
    fake = FakeCollector()
    monkeypatch.setattr(dashboard_observer, "_find_collector", lambda: fake)
    return fake


def _started(process: dict) -> DashboardObserver:
    observer = DashboardObserver()
    observer.start_process(dict(process))
    return observer


def _pushed(method: FakeRemoteMethod) -> list:
    return [record for (args, _) in method.calls for record in args[0]]


def _trajectory_event(ts: float) -> dict:
    return dict(ts=ts, kind="gen_start", sample_index=5, group_index=-1, turn=1, weight_version="2", detail="")


def test_a_run_the_dashboard_cannot_serve_is_refused_before_it_starts(monkeypatch):
    """Turning the dashboard on without ``--dump-details`` must stop the run, not let it train without telemetry."""
    monkeypatch.setenv("MILES_DASHBOARD", "1")
    installed = [
        EntryPoint("miles_dashboard", "miles.dashboard.observer:DashboardObserver", observers.ENTRY_POINT_GROUP)
    ]
    monkeypatch.setattr(observers, "entry_points", lambda group: installed)
    SingletonMeta._instances.pop(observers.Observers, None)
    args = SimpleNamespace(dump_details=None, cluster_backend="ray", use_rollout_entropy=True)

    with pytest.raises(observers.RunRefusedError, match="dump-details"):
        observers.start_run(run_uuid="0123456789abcdef", args=args)


def test_a_phase_start_is_pushed_at_once_as_an_open_phase(collector):
    """An open phase must reach the timeline while it runs, not when it ends."""
    observer = _started(TRAIN_RANK)

    observer.phase_start("rollout", 100.0)

    [(args, _)] = collector.push_phases.calls
    [event] = args[0]
    assert (event.name, event.t0, event.t1) == ("rollout", 100.0, PhaseEvent.OPEN_T1)
    assert event.open
    assert (event.node, event.gpus, event.rank, event.role) == ("10.0.0.3", [3], 7, Role.TRAIN)


def test_phase_ends_are_pushed_in_batches_by_count(collector, monkeypatch):
    """Closed phases cross to the collector in batches, carrying the identity the process started with."""
    monkeypatch.setattr(dashboard_observer, "BATCH_MAX_SECONDS", 3600.0)  # only the count can push here
    observer = _started(TRAIN_RANK)

    for i in range(BATCH_MAX_EVENTS - 1):
        observer.phase_end(f"phase_{i}", float(i), float(i) + 0.5)
    assert collector.push_phases.calls == []
    observer.phase_end("actor_train", 100.0, 160.0)

    [(args, _)] = collector.push_phases.calls
    [batch] = args
    assert len(batch) == BATCH_MAX_EVENTS
    event = batch[-1]
    assert (event.name, event.t0, event.t1) == ("actor_train", 100.0, 160.0)
    assert (event.node, event.gpus, event.rank, event.role) == ("10.0.0.3", [3], 7, Role.TRAIN)


def test_phase_ends_are_pushed_in_batches_by_age(collector, monkeypatch):
    """A slow trickle of phases must not wait for a full batch."""
    monkeypatch.setattr(dashboard_observer, "BATCH_MAX_SECONDS", 3600.0)
    observer = _started(TRAIN_RANK)

    observer.phase_end("a", 1.0, 2.0)
    assert collector.push_phases.calls == []
    observer._phases._last_push -= 3601.0
    observer.phase_end("b", 2.0, 3.0)

    assert [event.name for event in _pushed(collector.push_phases)] == ["a", "b"]


def test_a_process_that_went_quiet_still_pushes_its_last_phases(collector, monkeypatch):
    """The last phases of a rank idling through an eval must not stay open on the timeline."""
    monkeypatch.setattr(dashboard_observer, "BATCH_MAX_SECONDS", 0.5)
    observer = _started(TRAIN_RANK)

    observer.phase_end("tail", 1.0, 2.0)
    assert collector.push_phases.calls == []
    deadline = time.monotonic() + 10
    while not collector.push_phases.calls and time.monotonic() < deadline:
        time.sleep(0.02)

    assert [event.name for event in _pushed(collector.push_phases)] == ["tail"]


def test_only_train_ranks_and_the_rollout_manager_record_phases(collector):
    """The timeline has lanes for training ranks and the rollout manager only; other processes' phases are left out."""
    controller = _started(dict(role="inference_controller", node="10.0.0.4", rank=-1, gpus=[]))
    controller.phase_start("update_weights", 1.0)
    controller.phase_end("update_weights", 1.0, 2.0)
    controller._phases.flush()
    assert collector.push_phases.calls == []

    manager = _started(ROLLOUT_MANAGER)
    manager.phase_start("rollout", 3.0)

    [(args, _)] = collector.push_phases.calls
    [event] = args[0]
    assert (event.role, event.node, event.gpus, event.rank) == (Role.ROLLOUT_MANAGER, "10.0.0.5", [], -1)


def test_metrics_keep_only_scalars(collector):
    """The store persists plain scalars, so containers are dropped and numpy scalars become floats."""
    observer = _started(TRAIN_RANK)

    observer.metrics(
        {"a": 1.5, "b": "text", "c": [1, 2], "d": np.float32(2.5), "e": {"nested": 1}},
        step_key="rollout/step",
        step=3,
    )

    [(args, _)] = collector.push_metrics.calls
    [record] = args
    assert record.metrics == {"a": 1.5, "b": "text", "d": 2.5}
    assert (record.step, record.step_key) == (3, "rollout/step")


def test_each_event_reaches_its_collector_method(collector):
    """The data buffer, engines, sandboxes and sample lifecycles land as the records the store reads."""
    observer = _started(ROLLOUT_MANAGER)
    engine = dict(addr="http://a:1", worker_type="regular", engine_rank=0, gpus=[["10.0.0.5", 0]], gpu_uuids=[None])
    report = dict(ts=1.0, sandbox_key="sb", status="error", task=None, provider="docker", sandbox_id=None, error="x")

    observer.data_buffer(7)
    observer.engine_topology([engine])
    observer.sandbox_report(report)
    observer.trajectory_event(_trajectory_event(2.0))
    observer._trajectories.flush()

    [((buffer_sample,), _)] = collector.push_data_buffer.calls
    assert buffer_sample.length == 7
    [((snapshot,), _)] = collector.update_topology.calls
    assert snapshot.engines == [EngineInfo(**engine)]
    assert collector.push_sandbox_reports.calls == [(([sandbox_health.SandboxReport(**report)],), {})]
    assert _pushed(collector.push_trajectories) == [TrajectoryEvent(**_trajectory_event(2.0))]


def test_events_without_a_collector_are_dropped(monkeypatch):
    """A process that cannot find the collector must drop its events instead of failing on each one."""
    monkeypatch.setattr(dashboard_observer, "_find_collector", lambda: None)
    observer = _started(TRAIN_RANK)

    observer.phase_start("a", 1.0)
    observer.phase_end("a", 1.0, 2.0)
    observer.trajectory_event(_trajectory_event(1.0))
    observer.metrics({"a": 1.0}, step_key="rollout/step", step=0)
    observer.data_buffer(1)
    observer.engine_topology([])
    observer.sandbox_report(dict(ts=1.0, sandbox_key="sb", status="healthy"))
    observer.end_process()

    assert observer._phases._records == [] and observer._trajectories._records == []


def test_end_process_sends_what_the_process_holds_then_waits_until_it_landed(collector, monkeypatch):
    """A worker is killed right after ``end_process``, so a push still in flight would die with it."""
    sent = []
    monkeypatch.setattr(collector.push_phases, "remote", lambda batch: sent.append([event.name for event in batch]))
    monkeypatch.setattr(
        collector.push_trajectories, "remote", lambda batch: sent.append([event.kind for event in batch])
    )
    monkeypatch.setattr(collector.ping, "remote", lambda: "ping")
    monkeypatch.setattr(
        dashboard_observer, "ray", SimpleNamespace(get=lambda ref, timeout: sent.append((ref, timeout)))
    )
    observer = _started(ROLLOUT_MANAGER)
    observer.phase_end("eval", 1.0, 2.0)
    observer.trajectory_event(_trajectory_event(1.5))

    observer.end_process()

    assert sent == [["eval"], ["gen_start"], ("ping", END_PROCESS_TIMEOUT_SECONDS)]


def test_end_run_shuts_down_only_a_collector_it_created(collector, monkeypatch):
    """Only the driver, which created the collector, waits for its last flush and kills it."""
    ray_calls = []
    monkeypatch.setattr(
        dashboard_observer,
        "ray",
        SimpleNamespace(
            get=lambda ref, timeout: ray_calls.append(("get", timeout)),
            kill=lambda handle: ray_calls.append(("kill", handle)),
        ),
    )
    worker = _started(TRAIN_RANK)

    worker.end_run(succeeded=True)

    assert collector.shutdown.calls == [] and ray_calls == []

    driver = DashboardObserver()
    driver._collector, driver._owns_collector = collector, True
    driver.end_run(succeeded=True)

    assert len(collector.shutdown.calls) == 1
    assert ray_calls == [("get", SHUTDOWN_TIMEOUT_SECONDS), ("kill", collector)]


def test_trajectory_events_round_trip_through_the_store(collector, tmp_path):
    """Plain lifecycle events must load back from the store, each in its hour partition and by time window."""
    observer = _started(ROLLOUT_MANAGER)
    for ts in (10.0, 3700.0):
        observer.trajectory_event(_trajectory_event(ts))
    observer._trajectories.flush()

    writer = MetricStore(tmp_path)
    for event in _pushed(collector.push_trajectories):
        writer.append(event)
    writer.flush()
    store = MetricStore.load(tmp_path)

    assert store.has_stream(Stream.TRAJECTORIES)
    assert len(list((tmp_path / "trajectories").glob("*.jsonl"))) == 2
    assert [e.ts for e in store.trajectory_events()] == [10.0, 3700.0]
    assert [e.ts for e in store.trajectory_events(t0=3600.0)] == [3700.0]
    assert store.iter_records(Stream.TRAJECTORIES)[0].weight_version == "2"
