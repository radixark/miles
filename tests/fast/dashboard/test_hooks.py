import logging

import numpy as np
import pytest

from miles.dashboard import backend, hooks
from miles.dashboard.hooks import BATCH_MAX_EVENTS, BATCH_MAX_SECONDS, _Identity
from miles.dashboard.store import Role
from miles.utils.timer import Timer


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


class FakeEngineHandle:
    def __init__(self, info):
        self._info = info
        self.get_topology_info = self

    def remote(self):
        return self._info  # hooks._ray_get is patched to the identity function


class FakeServerEngine:
    def __init__(self, info, alive=True):
        self.actor_handle = FakeEngineHandle(info)
        self.is_allocated = alive
        self.is_alive = alive


class FakeGroup:
    def __init__(self, engines, nodes_per_engine=1):
        self.all_engines = engines
        self.nodes_per_engine = nodes_per_engine


def _info(url, node, gpus):
    return dict(url=url, node_ip=node, gpu_ids=gpus, gpu_uuids=[None] * len(gpus), worker_type="regular", node_rank=0)


def _servers(*groups):
    server = type("FakeServer", (), {"server_groups": list(groups)})()
    return {"default": server}


def test_register_engines_groups_multinode_and_dedups(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    # one multi-node engine (master + worker node) and one single-node engine
    master = FakeServerEngine(_info("http://a:1", "node-a", [0, 1]))
    worker = FakeServerEngine(_info("http://a-worker:1", "node-b", [0, 1]))
    single = FakeServerEngine(_info("http://b:1", "node-a", [2, 3]))
    servers = _servers(FakeGroup([master, worker], nodes_per_engine=2), FakeGroup([single]))

    hooks.register_engines(servers)
    [(args, _)] = handle.update_topology.calls
    [snapshot] = args
    assert [e.addr for e in snapshot.engines] == ["http://a:1", "http://b:1"]
    multinode = snapshot.engines[0]
    assert multinode.gpus == [["node-a", 0], ["node-a", 1], ["node-b", 0], ["node-b", 1]]

    hooks.register_engines(servers)  # steady state: no remote traffic
    assert len(handle.update_topology.calls) == 1

    single.actor_handle = FakeEngineHandle(_info("http://b:2", "node-a", [2, 3]))  # recovery: new actor
    hooks.register_engines(servers)
    assert len(handle.update_topology.calls) == 2
    assert handle.update_topology.calls[-1][0][0].engines[1].addr == "http://b:2"


def test_register_engines_skips_dead_chunks(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    alive = FakeServerEngine(_info("http://a:1", "n", [0]))
    dead = FakeServerEngine(_info("http://b:1", "n", [1]), alive=False)
    hooks.register_engines(_servers(FakeGroup([alive]), FakeGroup([dead])))

    [(args, _)] = handle.update_topology.calls
    assert [e.addr for e in args[0].engines] == ["http://a:1"]


def test_register_engines_without_collector_is_noop():
    hooks.register_engines(_servers(FakeGroup([FakeServerEngine(_info("http://a:1", "n", [0]))])))
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


def _router_args(ip="10.0.0.5", port=3333):
    return type("Args", (), {"sglang_router_ip": ip, "sglang_router_port": port, "use_miles_router": False})()


def test_register_router_pushes_resolved_addr(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.register_router(_router_args())
    [(args, kwargs)] = handle.set_router.calls
    assert args == ("http://10.0.0.5:3333",)
    assert kwargs == {"use_miles_router": False}


def test_register_router_before_router_start_is_a_wiring_bug(monkeypatch):
    monkeypatch.setattr(backend, "_handle", FakeHandle())
    with pytest.raises(AssertionError, match="after start_rollout_servers"):
        hooks.register_router(_router_args(ip=None))


def test_register_router_without_collector_is_noop():
    hooks.register_router(_router_args())  # must not raise


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


def test_phase_sink_begin_flushes_previous_completion_before_open_event():
    from miles.dashboard.store import PhaseEvent

    handle = FakeHandle()
    hooks.attach_phase_sink(handle, Role.TRAIN)
    [sink] = Timer().event_sinks

    sink("data_preprocess", 100.0, 100.4)
    assert handle.push_phases.calls == []

    sink.begin("critic_train", 100.4)
    [(args, _)] = handle.push_phases.calls
    completed, opened = args[0]
    assert (completed.name, completed.t0, completed.t1) == ("data_preprocess", 100.0, 100.4)
    assert (opened.name, opened.t0, opened.t1) == ("critic_train", 100.4, PhaseEvent.OPEN_T1)
