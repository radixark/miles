import logging

import numpy as np
import pytest

from miles.dashboard import backend, hooks
from miles.dashboard.hooks import BATCH_MAX_EVENTS, BATCH_MAX_SECONDS, _Identity
from miles.dashboard.store import Role
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.utils.timer import Timer
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


class FakeRemoteMethod:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def remote(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("collector unreachable")
        self.calls.append((args, kwargs))


class FakeHandle:
    def __init__(self, fail_push=False):
        self.push_phases = FakeRemoteMethod(fail=fail_push)
        self.push_metrics = FakeRemoteMethod()
        self.update_topology = FakeRemoteMethod()
        self.set_router = FakeRemoteMethod()
        self.push_data_buffer = FakeRemoteMethod()


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    timer = Timer()
    saved = list(timer.event_sinks)
    timer.event_sinks.clear()
    monkeypatch.setattr(hooks, "_phase_sink", None)
    monkeypatch.setattr(hooks, "_engines_fingerprint", None)
    monkeypatch.setattr(hooks, "_resolve_identity", lambda: _Identity(node="10.0.0.3", gpus=[3], rank=7))
    monkeypatch.setattr(hooks, "_ray_get", lambda refs: refs)
    monkeypatch.setattr(backend, "_handle", None)
    monkeypatch.setattr(backend, "_is_primary", False)
    monkeypatch.setattr(backend, "_resolution_failed", False)
    yield
    timer.event_sinks[:] = saved


# ------------------------------- phase sink ---------------------------------


def test_phase_sink_batches_by_count():
    handle = FakeHandle()
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks

    for i in range(BATCH_MAX_EVENTS - 1):
        sink(f"phase_{i}", float(i), float(i) + 0.5)
    assert handle.push_phases.calls == []

    sink("actor_train", 100.0, 160.0)
    [(args, _)] = handle.push_phases.calls
    [batch] = args
    assert len(batch) == BATCH_MAX_EVENTS
    event = batch[-1]
    assert (event.name, event.t0, event.t1) == ("actor_train", 100.0, 160.0)
    assert (event.node, event.gpus, event.rank, event.role) == ("10.0.0.3", [3], 7, Role.TRAIN)


def test_phase_sink_batches_by_time():
    handle = FakeHandle()
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks

    sink("a", 1.0, 2.0)
    assert handle.push_phases.calls == []
    sink._last_flush -= BATCH_MAX_SECONDS + 1
    sink("b", 2.0, 3.0)
    [(args, _)] = handle.push_phases.calls
    assert [e.name for e in args[0]] == ["a", "b"]


def test_phase_sink_reresolves_until_rank_known(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(hooks, "_resolve_identity", lambda: _Identity(node="n", gpus=[0], rank=-1))
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks

    sink("early", 1.0, 2.0)  # torch.distributed not initialized yet
    monkeypatch.setattr(hooks, "_resolve_identity", lambda: _Identity(node="n", gpus=[0], rank=5))
    sink("late", 2.0, 3.0)
    hooks.detach_and_flush()

    [(args, _)] = handle.push_phases.calls
    assert [event.rank for event in args[0]] == [-1, 5]


def test_phase_sink_swallows_push_failures(caplog):
    handle = FakeHandle(fail_push=True)
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks
    with caplog.at_level(logging.WARNING):
        for i in range(BATCH_MAX_EVENTS):
            sink("p", float(i), float(i) + 1)  # must not raise into Timer.end()
    assert any("phase sink failed" in r.message for r in caplog.records)


def test_attach_is_idempotent_and_detach_flushes():
    handle = FakeHandle()
    hooks.attach_phase_sink(handle, Role.TRAIN)
    hooks.attach_phase_sink(handle, Role.ROLLOUT_MANAGER)  # second attach ignored
    assert len(Timer().event_sinks) == 1

    Timer().event_sinks[0]("tail", 1.0, 2.0)
    hooks.detach_and_flush()
    assert Timer().event_sinks == []
    [(args, _)] = handle.push_phases.calls
    assert [e.name for e in args[0]] == ["tail"]


def test_register_train_actor_disabled_is_free(monkeypatch):
    monkeypatch.setattr(backend, "resolve_collector", lambda: pytest.fail("must not resolve when disabled"))
    hooks.register_train_actor(type("Args", (), {"use_miles_dashboard": False})())
    assert Timer().event_sinks == []


def test_register_train_actor_attaches_train_sink(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "resolve_collector", lambda: handle)
    hooks.register_train_actor(type("Args", (), {"use_miles_dashboard": True})())
    [sink] = Timer().event_sinks
    assert sink.role == Role.TRAIN


# ---------------------------- engine registration ---------------------------


class _FakeProbe:
    def __init__(self, value_fn):
        self._value_fn = value_fn

    def remote(self, *args, **kwargs):
        return self._value_fn(*args, **kwargs)  # hooks._ray_get is patched to the identity function


class FakeEngineHandle:
    def __init__(self):
        self._get_gpu_uuids = _FakeProbe(lambda gpu_ids: [None] * len(gpu_ids))


class FakeManagerHandle:
    """Duck-typed RayWorkerManager handle serving per-cell worker infos."""

    def __init__(self, infos_by_cell):
        self.get_worker_infos = _FakeProbe(lambda *, pool, cell_index: infos_by_cell[(pool, cell_index)])


class FakeCell:
    """Duck-typed ServerCell: the hooks read only the driver-side routing facts."""

    def __init__(self, url, cell_index=0, alive=True):
        self.meta = ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id=f"inference-engine-0-0-{cell_index}",
            num_gpus_per_engine=1,
            gpu_offset=cell_index,
            sglang_api_key=None,
            worker_name=f"inference-engine-0-0-{cell_index}-0",
            needs_offload=False,
            update_weights=False,
            workers_hash=f"hash-{cell_index}",
        )
        self.server_url = url
        self.is_pending_weights_or_serving = alive


def _worker_info(name, node, gpus, generation=1):
    return WorkerInfo(
        name=name,
        generation=generation,
        self_addrs={"primary": HostAndPort(host=node, port=30001)},
        gpu_ids=gpus,
        actor_handle=FakeEngineHandle(),
    )


def _servers(cells):
    server = type("FakeServer", (), {"server_cells": {f"cell-{i}": cell for i, cell in enumerate(cells)}})()
    return {"default": server}


def test_register_engines_groups_multinode_and_dedups(monkeypatch):
    """Worker-manager infos become one EngineInfo per cell; repush only on worker change."""
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    infos_by_cell = {
        ("inference-engine-0-0", 0): [
            _worker_info("inference-engine-0-0-0", "node-a", [0, 1]),
            _worker_info("inference-engine-0-0-1", "node-b", [0, 1]),
        ],
        ("inference-engine-0-0", 1): [_worker_info("inference-engine-0-1-0", "node-a", [2, 3])],
    }
    monkeypatch.setattr(RayWorkerManager, "get_handle", staticmethod(lambda: FakeManagerHandle(infos_by_cell)))
    servers = _servers([FakeCell("http://a:1", cell_index=0), FakeCell("http://b:1", cell_index=1)])

    hooks.register_engines(servers)
    [(args, _)] = handle.update_topology.calls
    [snapshot] = args
    assert [e.addr for e in snapshot.engines] == ["http://a:1", "http://b:1"]
    multinode = snapshot.engines[0]
    assert multinode.gpus == [["node-a", 0], ["node-a", 1], ["node-b", 0], ["node-b", 1]]
    assert len(multinode.gpu_uuids) == 4

    hooks.register_engines(servers)  # steady state: fingerprint unchanged
    assert len(handle.update_topology.calls) == 1

    infos_by_cell[("inference-engine-0-0", 1)] = [
        _worker_info("inference-engine-0-1-0", "node-a", [2, 3], generation=2)
    ]
    hooks.register_engines(servers)  # recovery: same worker, new generation
    assert len(handle.update_topology.calls) == 2


def test_register_engines_skips_dead_cells(monkeypatch):
    """Cells that are not alive are left out of the snapshot and never queried."""
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    infos_by_cell = {("inference-engine-0-0", 0): [_worker_info("inference-engine-0-0-0", "n", [0])]}
    monkeypatch.setattr(RayWorkerManager, "get_handle", staticmethod(lambda: FakeManagerHandle(infos_by_cell)))

    hooks.register_engines(
        _servers([FakeCell("http://a:1", cell_index=0), FakeCell("http://b:1", cell_index=1, alive=False)])
    )

    [(args, _)] = handle.update_topology.calls
    assert [e.addr for e in args[0].engines] == ["http://a:1"]


def test_register_engines_survives_missing_worker_manager(monkeypatch, caplog):
    """Engines not yet owned by the worker manager degrade to a warning, not a crash."""
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)

    def _no_manager():
        raise ValueError("worker manager actor not found")

    monkeypatch.setattr(RayWorkerManager, "get_handle", staticmethod(_no_manager))
    hooks._warner.reset_window_for_test()
    with caplog.at_level(logging.WARNING):
        hooks.register_engines(_servers([FakeCell("http://a:1")]))

    assert handle.update_topology.calls == []
    assert any("engine registration failed" in r.message for r in caplog.records)


def test_register_engines_without_collector_is_noop():
    hooks.register_engines(_servers([FakeCell("http://a:1")]))
    assert hooks._engines_fingerprint is None


# ------------------------------ dashboard_log -------------------------------


def test_dashboard_log_filters_to_scalars(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    backend.dashboard_log(
        {"a": 1.5, "b": "text", "c": [1, 2], "d": np.float32(2.5), "e": {"nested": 1}},
        step=3,
        step_key="rollout/step",
    )
    [(args, _)] = handle.push_metrics.calls
    [record] = args
    assert record.metrics == {"a": 1.5, "b": "text", "d": 2.5}
    assert record.step == 3 and record.step_key == "rollout/step"


def test_dashboard_log_without_handle_is_noop():
    backend.dashboard_log({"a": 1})  # must not raise


# ----------------------------- router registration --------------------------


def _router_args(ip="10.0.0.5", port=3333, use_miles_dashboard=True):
    return type(
        "Args",
        (),
        {
            "sglang_router_ip": ip,
            "sglang_router_port": port,
            "use_miles_dashboard": use_miles_dashboard,
        },
    )()


def test_register_router_pushes_resolved_addr(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.register_router(_router_args())
    [(args, kwargs)] = handle.set_router.calls
    assert args == ("http://10.0.0.5:3333",)
    assert kwargs == {}


def test_register_router_resolves_the_collector_itself(monkeypatch):
    """register_router works in a process that never ran init_tracking."""
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", None)
    monkeypatch.setattr(backend, "resolve_collector", lambda: handle)
    hooks.register_router(_router_args())
    assert len(handle.set_router.calls) == 1


def test_register_router_before_router_start_is_a_wiring_bug(monkeypatch):
    monkeypatch.setattr(backend, "_handle", FakeHandle())
    with pytest.raises(AssertionError, match="after start_rollout_servers"):
        hooks.register_router(_router_args(ip=None))


def test_register_router_without_dashboard_is_noop(monkeypatch):
    """With the dashboard off the hook returns before resolve_collector, which would block."""
    monkeypatch.setattr(backend, "resolve_collector", _never_resolve)
    hooks.register_router(_router_args(use_miles_dashboard=False))


def _never_resolve():
    raise AssertionError("resolve_collector must not be called when the dashboard is disabled")


# ---------------------------- data buffer report ----------------------------


def test_report_data_buffer_pushes_length(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.report_data_buffer(7)
    [(args, kwargs)] = handle.push_data_buffer.calls
    (sample,) = args
    assert sample.length == 7
    assert kwargs == {}


def test_report_data_buffer_none_is_noop(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.report_data_buffer(None)  # plain RolloutDataSource: nothing to report
    assert handle.push_data_buffer.calls == []


def test_report_data_buffer_without_collector_is_noop():
    hooks.report_data_buffer(7)  # must not raise


def test_report_data_buffer_swallows_push_failures(monkeypatch, caplog):
    handle = FakeHandle()
    handle.push_data_buffer = FakeRemoteMethod(fail=True)
    monkeypatch.setattr(backend, "_handle", handle)
    with caplog.at_level(logging.WARNING):
        hooks.report_data_buffer(7)  # must not raise
    assert any("data-buffer report failed" in r.message for r in caplog.records)


def test_phase_sink_begin_pushes_open_event_immediately():
    from miles.dashboard.store import PhaseEvent

    handle = FakeHandle()
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks

    sink.begin("rollout", 100.0)
    [(args, _)] = handle.push_phases.calls  # no batching for starts
    [event] = args[0]
    assert event.name == "rollout" and event.t0 == 100.0
    assert event.open and event.t1 == PhaseEvent.OPEN_T1
    assert (event.node, event.rank) == ("10.0.0.3", 7)
