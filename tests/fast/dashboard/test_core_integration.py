"""Integration of the dashboard with miles core, on a local Ray instance.

Exercises the REAL activation path: tracking registry -> MilesDashboardBackend
-> init_dashboard (named collector actor + samplers) -> tracking_utils.log
fan-out -> Timer phase sink -> finish_tracking, then loads the produced
directory with the store and serves it. GPU samples appear only on hosts with
NVML; everything else must work identically on a laptop and a GPU node.
"""

import time
from argparse import Namespace

import pytest

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, Role, Stream
from miles.utils.timer import Timer, timer


def _args(tmp_path) -> Namespace:
    return Namespace(
        use_miles_dashboard=True,
        dump_details=str(tmp_path),
        wandb_group="wiring-e2e",
        use_rollout_entropy=True,
        use_miles_router=False,
        dashboard_flush_interval=0.2,
        dashboard_gpu_sample_interval=0.2,
        dashboard_sglang_scrape_interval=2.0,
        dashboard_sglang_scrape_mode="auto",
        dashboard_sglang_metrics=None,
        dashboard_forward_prometheus=False,
    )


def test_tracking_backend_end_to_end_local_ray(tmp_path):
    ray = pytest.importorskip("ray")
    from miles.dashboard import backend, hooks
    from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking
    from miles.utils.tracking_utils.tracking import log as tracking_log

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
    saved_sinks = list(Timer().event_sinks)
    args = _args(tmp_path)
    try:
        # driver: registry picks MilesDashboardBackend from the args flag
        init_tracking(args)
        assert backend.current_collector() is not None

        # metric fan-out through the real tracking API
        tracking_log(args, {"rollout/rewards_mean": 0.5, "rollout/step": 3}, step_key="rollout/step")

        # per-rank phase lane path (driver doubles as a "rank" here)
        hooks.register_train_actor(args)
        with timer("actor_train"):
            time.sleep(0.05)
        with timer("update_weights"):
            time.sleep(0.01)

        finish_tracking()  # flushes sink + synchronous collector shutdown
    finally:
        Timer().event_sinks[:] = saved_sinks
        ray.shutdown()

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


def test_registry_contains_dashboard_backend():
    from miles.utils.tracking_utils.base import MilesDashboardBackend
    from miles.utils.tracking_utils.tracking import BACKEND_REGISTRY

    cls, flag = BACKEND_REGISTRY["miles_dashboard"]
    assert cls is MilesDashboardBackend
    assert flag == "use_miles_dashboard"


def test_engine_topology_gpu_range_logic():
    # the pure slice of the topology construction: node-physical gpu ranges
    pytest.importorskip("sglang")
    from miles.ray.rollout.server_cell import ServerCell

    cell = ServerCell(
        args=Namespace(num_gpus_per_node=8),
        worker_type="regular",
        cell_id="cell-0",
        pg=(None, [], [4, 5, 6, 7]),
        num_gpus_per_engine=2,
    )
    assert cell.engine_gpu_ids == [[4, 5]]

    # multi-node engine: each member covers its whole node (base 0, capped per node)
    multi_node = ServerCell(
        args=Namespace(num_gpus_per_node=8),
        worker_type="regular",
        cell_id="cell-0",
        pg=(None, [], list(range(8)) + list(range(8))),
        num_nodes=2,
        num_gpus_per_engine=16,
    )
    assert multi_node.engine_gpu_ids == [list(range(8)), list(range(8))]
