"""Integration of the dashboard with miles core, on a local Ray instance.

Exercises the REAL activation path: ``MILES_DASHBOARD=1`` turns on the
dashboard observer at ``start_run`` (named collector actor + samplers) ->
tracking_utils.log fan-out -> a train rank's Timer phases -> ``end_run``,
then loads the produced directory with the store and serves it. GPU samples
appear only on hosts with NVML; everything else must work identically on a
laptop and a GPU node.
"""

import re
import time
from argparse import Namespace
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.observer import DashboardObserver
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, Role, Stream
from miles.utils import observers
from miles.utils.timer import Timer, timer
from miles.utils.tracking_utils.tracking import log as tracking_log


def _args(tmp_path) -> Namespace:
    return Namespace(
        dump_details=str(tmp_path), wandb_group="wiring-e2e", use_rollout_entropy=True, cluster_backend="ray"
    )


def test_the_dashboard_observer_end_to_end_on_local_ray(tmp_path, ray_local_mode, monkeypatch):
    monkeypatch.setenv("MILES_DASHBOARD", "1")
    monkeypatch.setenv("MILES_DASHBOARD_FLUSH_INTERVAL", "0.2")
    monkeypatch.setenv("MILES_DASHBOARD_GPU_SAMPLE_INTERVAL", "0.2")
    installed = [
        EntryPoint("miles_dashboard", "miles.dashboard.observer:DashboardObserver", observers.ENTRY_POINT_GROUP)
    ]
    monkeypatch.setattr(observers, "entry_points", lambda group: installed)
    saved_sinks = list(Timer().event_sinks)
    args = _args(tmp_path)
    try:
        # driver: MILES_DASHBOARD turns the observer on, and it hands the switch to the workers
        assert observers.start_run(run_uuid="0123456789abcdef", args=args) == {"MILES_DASHBOARD": "1"}

        # metric fan-out through the real tracking API
        tracking_log(args, {"rollout/rewards_mean": 0.5, "rollout/step": 3}, step_key="rollout/step")

        # per-rank phase lane path (driver doubles as a "rank" here)
        observers.start_process("train", rank=0)
        with timer("actor_train"):
            time.sleep(0.05)
        with timer("update_weights"):
            time.sleep(0.01)

        observers.Observers().end_run(succeeded=True)  # flushes the phases + synchronous collector shutdown
    finally:
        Timer().event_sinks[:] = saved_sinks

    store = MetricStore.load(tmp_path / "dashboard")
    assert store.meta.run_name == "wiring-e2e"

    [metric] = store.records[Stream.METRICS]
    assert metric.step == 3 and metric.metrics["rollout/rewards_mean"] == 0.5

    phases = {event.name: event for event in store.iter_records(Stream.PHASES)}
    assert {"actor_train", "update_weights"} <= set(phases)
    actor_train = phases["actor_train"]
    assert actor_train.role == Role.TRAIN
    assert actor_train.t1 - actor_train.t0 == pytest.approx(0.05, abs=0.05)

    app_client = pytest.importorskip("fastapi.testclient").TestClient(
        make_app(store, DumpReader(tmp_path), follow=False)
    )
    meta = app_client.get("/api/meta").json()
    assert meta["run_name"] == "wiring-e2e"
    assert meta["capabilities"]["has_timeline"] is True
    assert "rollout/rewards_mean" in meta["metric_keys"]


def test_miles_registers_the_dashboard_observer():
    """miles finds the dashboard only through this entry point, so it must name the observer that feeds the collector."""
    setup_py = (Path(__file__).resolve().parents[3] / "setup.py").read_text()
    [target] = re.findall(r'"miles_dashboard = ([\w.:]+)"', setup_py)
    assert EntryPoint("miles_dashboard", target, observers.ENTRY_POINT_GROUP).load() is DashboardObserver
