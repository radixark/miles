"""Job-scoped worker teardown with real lifecycle code and no Ray/GPU processes.

Run with --noconftest -p pytest_asyncio.plugin to avoid optional GPU imports in
repository-wide fixtures. All process signaling and Ray actor operations are fakes.
"""

import asyncio
import importlib.util
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule


class DummyTrainer:
    pass


@pytest.fixture
def worker_lifecycle(monkeypatch):
    root = Path(__file__).resolve().parents[4]

    def install(name, module):
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        monkeypatch.setitem(sys.modules, name, module)
        if len(parts) > 1:
            monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], parts[-1], module, raising=False)

    stubs = {
        "ray": {},
        "ray.util.scheduling_strategies": {"PlacementGroupSchedulingStrategy": lambda **kw: SimpleNamespace(**kw)},
        "miles.utils.http_utils": {"_wrap_ipv6": lambda ip: f"[{ip}]" if ":" in ip else ip},
        "miles.utils.ray_utils": {"compute_ray_pin_head_options": lambda: {}},
        "miles.utils.misc": {"NodeProbeMixin": object},
        "miles.utils.test_utils.fault_injector": {"inject_fault": Mock()},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        install(name, module)

    modules = {}
    for name in (
        "miles.utils.pydantic_utils",
        "miles.utils.function_registry",
        "miles.utils.workers.naming",
        "miles.utils.workers.worker_spec",
        "miles.utils.workers.worker_handle",
        "miles.utils.workers.worker_info",
        "miles.utils.workers.worker_provider.base",
        "miles.utils.workers.ray_worker_handle",
        "miles.utils.workers.addr_allocator",
        "miles.utils.workers.process_utils",
        "miles.utils.workers.command_actor",
        "miles.utils.workers.ray_worker_manager",
    ):
        spec = importlib.util.spec_from_file_location(name, root / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        install(name, module)
        spec.loader.exec_module(module)
        modules[name.rsplit(".", 1)[-1]] = module
    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(modules["ray_worker_manager"], "ray", fake_ray)
    return SimpleNamespace(**modules, cluster=cluster, fake_ray=fake_ray)


def _spec(env, name, *, cells=1, workers=1, serve=False):
    kwargs = dict(
        name=name,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=env.worker_spec.SchedulingSpec(
            num_cells=cells, num_workers_per_cell=workers, num_gpus_per_worker=0
        ),
    )
    if serve:
        return env.worker_spec.ServeWorkerSpec(
            **kwargs, worker_class=f"{__name__}.DummyTrainer", ctor_kwargs=lambda _ctx: {}
        )
    return env.worker_spec.CommandWorkerSpec(**kwargs, launch_command=lambda _ctx: "never-executed-test-command")


async def test_dispose_stops_every_owned_pool_but_not_another_manager(worker_lifecycle):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    other = env.ray_worker_manager.RayWorkerManager()
    await manager.init(
        [
            _spec(env, "engine", cells=2, workers=2),
            _spec(env, "router"),
            _spec(env, "session"),
            _spec(env, "actor", workers=2, serve=True),
            _spec(env, "critic", serve=True),
            _spec(env, "disabled-external-placeholder", cells=0),
        ],
        {},
    )
    owned = list(env.cluster.handles)
    await other.init([_spec(env, "engine")], {})
    unrelated = env.cluster.handles[-1]

    await manager.dispose()
    await manager.dispose()

    assert len(owned) == 9
    assert all(handle.killed for handle in owned)
    assert not unrelated.killed
    assert env.cluster.events.count("kill") == len(owned)
    # Command workers shut down their process groups; serve/trainer workers
    # are terminated directly, because they do not implement shutdown RPCs.
    assert len(env.cluster.calls_of("shutdown")) == 6
    assert not any(cell.alive for cell in manager._all_cells())
    with pytest.raises(RuntimeError, match="closing"):
        await manager.start_cells(["engine-0"])


async def test_dispose_before_init_is_safe_and_terminal(worker_lifecycle):
    manager = worker_lifecycle.ray_worker_manager.RayWorkerManager()
    await manager.dispose()
    await manager.dispose()
    with pytest.raises(RuntimeError, match="closing"):
        await manager.init([_spec(worker_lifecycle, "engine")], {})
    assert worker_lifecycle.cluster.handles == []


async def test_dispose_cancels_partial_start_and_prevents_late_resurrection(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    started = asyncio.Event()

    async def blocked_alloc(_self):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(env.ray_worker_manager._BaseActorManager, "alloc_ports", blocked_alloc)
    init = asyncio.create_task(manager.init([_spec(env, "engine", workers=2)], {}))
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(manager.dispose(), 1)
    with pytest.raises(asyncio.CancelledError):
        await init
    assert all(handle.killed for handle in env.cluster.handles)
    assert not any(cell.alive for cell in manager._all_cells())
    assert env.cluster.calls_of("run") == []


async def test_failed_launch_waits_for_late_sibling_before_rollback(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    failed = asyncio.Event()
    original_launch = env.ray_worker_manager._CommandActorManager.launch_actor

    async def staggered_launch(self):
        if self.worker_in_cell_index == 0:
            failed.set()
            raise RuntimeError("first worker failed")
        await failed.wait()
        await asyncio.sleep(0.01)
        await original_launch(self)

    monkeypatch.setattr(env.ray_worker_manager._CommandActorManager, "launch_actor", staggered_launch)
    with pytest.raises(RuntimeError, match="first worker failed"):
        await manager.init([_spec(env, "engine", workers=2)], {})
    assert len(env.cluster.handles) == 1
    assert env.cluster.handles[0].killed
    assert not manager._pools["engine"].cells[0].alive


async def test_cancelled_init_rolls_back_all_started_actors(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    error = asyncio.CancelledError("init cancelled")

    async def cancelled_post(_self):
        raise error

    monkeypatch.setattr(env.ray_worker_manager._CommandActorManager, "post_setup", cancelled_post)
    with pytest.raises(asyncio.CancelledError):
        await manager.init([_spec(env, "engine", workers=2)], {})
    assert all(handle.killed for handle in env.cluster.handles)


async def test_failing_worker_does_not_short_circuit_other_cells_and_can_retry(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    await manager.init([_spec(env, "engine", cells=2, workers=2)], {})
    failing = env.cluster.handles[0]
    original_kill = env.fake_ray.kill
    failed_once = False

    def kill(handle, *, no_restart):
        nonlocal failed_once
        assert no_restart is True
        if handle is failing and not failed_once:
            failed_once = True
            raise RuntimeError("kill failed")
        original_kill(handle, no_restart=no_restart)

    monkeypatch.setattr(env.fake_ray, "kill", kill)
    with pytest.raises(RuntimeError, match="kill failed"):
        await manager.dispose()
    assert all(handle.killed for handle in env.cluster.handles[1:])
    assert manager._pools["engine"].cells[0].actors[0].actor_handle is failing
    await manager.dispose()
    assert all(handle.killed for handle in env.cluster.handles)
    assert not any(cell.alive for cell in manager._all_cells())


@pytest.mark.parametrize("shutdown_failure", ["error", "cancel", "timeout"])
async def test_shutdown_failure_still_kills_exact_actor(worker_lifecycle, monkeypatch, shutdown_failure):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    await manager.init([_spec(env, "engine", workers=2)], {})
    handle = env.cluster.handles[0]
    if shutdown_failure == "timeout":
        handle.hanging_methods["shutdown"] = 3600
        monkeypatch.setattr(env.ray_worker_manager, "_SHUTDOWN_TIMEOUT", 0.01)
    else:
        handle.failing_methods["shutdown"] = (
            asyncio.CancelledError() if shutdown_failure == "cancel" else RuntimeError("unavailable")
        )
    if shutdown_failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await manager.dispose()
    else:
        await manager.dispose()
    assert all(handle.killed for handle in env.cluster.handles)
    for handle in env.cluster.handles:
        assert any(call.handle is handle for call in env.cluster.calls_of("shutdown"))


@pytest.mark.parametrize("rollback_failed", [False, True])
def test_launch_failure_rolls_back_only_new_handle_and_preserves_error(worker_lifecycle, rollback_failed):
    env = worker_lifecycle
    unrelated = env.fake_ray.remote(DummyTrainer).remote()
    error = RuntimeError("init failed")
    env.cluster.method_errors["init"] = error
    if rollback_failed:
        env.cluster.method_errors["dispose"] = ValueError("rollback failed")
    with pytest.raises(RuntimeError) as caught:
        env.ray_worker_manager.RayWorkerManager.launch([], {})
    assert caught.value is error
    owned = env.cluster.handles[-1]
    assert env.cluster.calls_of("dispose")[0].handle is owned
    assert owned.killed is (not rollback_failed)
    assert not unrelated.killed
    assert env.cluster.get_timeouts[-1] == 60


def test_name_collision_never_attaches_to_or_stops_existing_manager(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    unrelated = env.fake_ray.remote(DummyTrainer).remote()
    error = ValueError("actor name already exists")
    monkeypatch.setattr(env.cluster, "create_actor", Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        env.ray_worker_manager.RayWorkerManager.launch([], {})
    assert caught.value is error
    assert not unrelated.killed
    assert env.cluster.calls == []


async def test_reinitialization_cannot_replace_owned_handles(worker_lifecycle):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    await manager.init([_spec(env, "engine")], {})
    with pytest.raises(RuntimeError, match="already initialized"):
        await manager.init([_spec(env, "replacement")], {})
    await manager.dispose()
    assert len(env.cluster.handles) == 1
    assert env.cluster.handles[0].killed


async def test_concurrent_dispose_does_not_interrupt_startup_rollback(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    manager = env.ray_worker_manager.RayWorkerManager()
    started = asyncio.Event()

    async def blocked_alloc(_self):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(env.ray_worker_manager._BaseActorManager, "alloc_ports", blocked_alloc)
    init = asyncio.create_task(manager.init([_spec(env, "engine", workers=2)], {}))
    await asyncio.wait_for(started.wait(), 1)
    env.cluster.handles[0].hanging_methods["shutdown"] = 0.01
    await asyncio.wait_for(asyncio.gather(manager.dispose(), manager.dispose()), 1)
    with pytest.raises(asyncio.CancelledError):
        await init
    assert all(handle.killed for handle in env.cluster.handles)
    assert env.cluster.events.count("kill") == 2


def test_command_shutdown_uses_its_owned_process_group_not_names(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    command = env.command_actor.CommandActor()
    process = Mock(pid=123456)
    command._process = process
    group_stop = Mock()
    monkeypatch.setattr(env.process_utils, "terminate_process_tree", group_stop)
    command.shutdown()
    group_stop.assert_called_once_with(process)
    assert command._shutting_down


def test_existing_process_group_teardown_signals_only_recorded_group(worker_lifecycle, monkeypatch):
    env = worker_lifecycle
    process = Mock(pid=123456)
    killpg = Mock()
    monkeypatch.setattr(env.process_utils.os, "killpg", killpg)
    env.process_utils.terminate_process_tree(process)
    assert [call.args for call in killpg.call_args_list] == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait.call_count == 2
