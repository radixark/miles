from __future__ import annotations

import functools
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from ray import cloudpickle
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers import ray_worker_manager as rwm
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _build_serve_worker, bootstrapped_worker_class
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec, WorkerLaunchContext

pytestmark = pytest.mark.asyncio


def _passthrough(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class _GroupedWorker:
    @rpc(concurrency_group="kill_self")
    def isolated(self) -> None: ...

    @rpc(concurrency_group="fault_injector")
    @_passthrough
    def wrapped_isolated(self) -> None: ...

    @rpc(concurrency_group="kill_self")
    @_passthrough
    @rpc(concurrency_group="heartbeat_status")
    def outer_declaration_wins(self) -> None: ...

    def plain(self) -> None: ...


class DemoServeWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_WORKER_CLASS_PATH = f"{DemoServeWorker.__module__}.{DemoServeWorker.__qualname__}"
_GROUPED_WORKER_CLASS_PATH = f"{_GroupedWorker.__module__}.{_GroupedWorker.__qualname__}"

_REPO_ROOT = Path(__file__).resolve().parents[4]

_REBUILD_IN_CHILD = """
import json
import sys

from ray import cloudpickle

from tests.fast.utils.workers.test_ray_worker_manager_serve import DemoServeWorker

actor_class, ctor_kwargs, context = cloudpickle.loads(sys.stdin.buffer.read())
worker = actor_class(ctor_kwargs=ctor_kwargs, context=context)
print(json.dumps(dict(**worker.kwargs, rebuilt=actor_class is not DemoServeWorker, name=actor_class.__name__)))
"""


def _make_spec(
    name: str = "trainer",
    *,
    num_cells: int = 1,
    num_workers_per_cell: int = 1,
    ctor_kwargs=None,
    concurrency_groups: dict[str, int] | None = None,
    num_gpus_per_worker: float = 0,
    num_cpus_per_worker: float = 0.2,
    num_gpu_slots_per_worker: int = 0,
    pg_name: str | None = None,
    env_var=None,
    worker_class: str = _WORKER_CLASS_PATH,
) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=env_var if env_var is not None else (lambda _ctx: {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
            num_cpus_per_worker=num_cpus_per_worker,
            num_gpu_slots_per_worker=num_gpu_slots_per_worker,
            pg_name=pg_name,
        ),
        worker_class=worker_class,
        ctor_kwargs=ctor_kwargs if ctor_kwargs is not None else (lambda _ctx: {}),
        concurrency_groups=concurrency_groups,
    )


@dataclass
class _CtorKwargsProbe:
    contexts: list[Any] = field(default_factory=list)

    def __call__(self, context: Any) -> dict[str, Any]:
        self.contexts.append(context)
        return dict(rank=context.worker_in_cell_index, role="actor")


class _RecordingCapability(BackendCapability):
    def __init__(self) -> None:
        self.operations = object()
        self.requested_pool_ids: list[list[str]] = []
        self.requested_static_pool_ids: list[str] = []

    def dynamic_worker_provider(self, *, pool_ids):
        self.requested_pool_ids.append(list(pool_ids))
        return object()

    def static_worker_provider(self, *, pool_id: str):
        self.requested_static_pool_ids.append(pool_id)
        return object()

    def cell_operations(self):
        return self.operations


def _launch_context(*, worker_in_cell_index: int = 0) -> WorkerLaunchContext:
    return WorkerLaunchContext(cell_index=0, worker_in_cell_index=worker_in_cell_index, gpu_ids=[])


def _make_pgs(num_slots: int = 8) -> dict[str, PlacementGroupInfo]:
    return {
        "actor": PlacementGroupInfo(
            pg="fake-pg",
            pg_reordered_bundle_indices=list(range(num_slots)),
            pg_reordered_gpu_ids=list(range(num_slots)),
        )
    }


async def _launch(specs, pgs=None, comm_backend: WorkerCommBackend = WorkerCommBackend.RAY) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, pgs if pgs is not None else {}, comm_backend=comm_backend)
    return manager


def _actor_classes(cluster: FakeRayCluster) -> list[type]:
    return [handle.actor_class for handle in cluster.handles]


def _options(cluster: FakeRayCluster) -> list[dict]:
    return [handle.options for handle in cluster.handles]


class TestServeWorkersAreLaunched:
    async def test_the_declared_worker_class_is_instantiated(self, fake_ray_cluster: FakeRayCluster):
        """A serve spec names its worker class instead of running a shell command."""
        await _launch([_make_spec(num_workers_per_cell=2)])

        assert [issubclass(cls, DemoServeWorker) for cls in _actor_classes(fake_ray_cluster)] == [True, True]

    async def test_no_launch_command_is_ever_sent(self, fake_ray_cluster: FakeRayCluster):
        """Serve workers start with their constructor, so post_setup must stay silent."""
        await _launch([_make_spec()])

        assert fake_ray_cluster.calls_of("run") == []

    async def test_the_manager_never_evaluates_the_ctor_kwargs_of_a_spec(self, fake_ray_cluster: FakeRayCluster):
        """ctor kwargs may hold a live provider, which cannot be shipped from here to the actor."""

        def explode(_ctx) -> dict:
            raise AssertionError("ctor kwargs were computed in the manager process")

        await _launch([_make_spec(num_workers_per_cell=2, ctor_kwargs=explode)])

        assert len(fake_ray_cluster.handles) == 2

    async def test_each_worker_is_handed_the_spec_s_ctor_kwargs_fn_and_its_own_launch_context(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """What ships is the recipe and the rank's identity; the evaluated kwargs never leave the actor."""
        probe = _CtorKwargsProbe()
        await _launch([_make_spec(num_workers_per_cell=3, ctor_kwargs=probe)])

        assert [list(kwargs) for kwargs in fake_ray_cluster.ctor_kwargs] == [["ctor_kwargs", "context"]] * 3
        assert all(kwargs["ctor_kwargs"] is probe for kwargs in fake_ray_cluster.ctor_kwargs)
        assert [kwargs["context"].worker_in_cell_index for kwargs in fake_ray_cluster.ctor_kwargs] == [0, 1, 2]

    async def test_gpu_ids_reach_the_actor(self, fake_ray_cluster: FakeRayCluster):
        """A serve worker cannot ask ray for its slot, so the manager must tell it."""
        spec = _make_spec(
            num_workers_per_cell=2,
            num_gpu_slots_per_worker=1,
            num_gpus_per_worker=0.4,
            pg_name="actor",
        )

        await _launch([spec], _make_pgs())

        assert [kwargs["context"].gpu_ids for kwargs in fake_ray_cluster.ctor_kwargs] == [[0], [1]]

    async def test_env_vars_are_computed_per_worker(self, fake_ray_cluster: FakeRayCluster):
        """Per-rank paths such as the offload directory live in the runtime env."""
        spec = _make_spec(num_workers_per_cell=2, env_var=lambda ctx: {"RANK_DIR": f"/d/{ctx.worker_in_cell_index}"})

        await _launch([spec])

        env_vars = [options["runtime_env"]["env_vars"] for options in _options(fake_ray_cluster)]
        assert env_vars == [{"RANK_DIR": "/d/0"}, {"RANK_DIR": "/d/1"}]


class TestServeWorkerClassFailures:
    async def test_an_unloadable_worker_class_rolls_back_the_serve_cell(self, fake_ray_cluster: FakeRayCluster):
        """A cell left alive around a class that cannot be imported would never be retried nor serve."""
        spec = _make_spec(worker_class=f"{_WORKER_CLASS_PATH}Missing")
        manager = RayWorkerManager()

        with pytest.raises(Exception, match="DemoServeWorkerMissing"):
            await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

        assert fake_ray_cluster.handles == []
        assert not manager.get_cell_infos(pool_ids=["trainer"])["trainer-00000"].alive


class TestServeSchedulingOptions:
    async def test_concurrency_groups_reach_ray(self, fake_ray_cluster: FakeRayCluster):
        """The trainer heartbeat rpc must not queue behind a running train step."""
        groups = {"heartbeat_status": 1, "default": 1}

        await _launch([_make_spec(concurrency_groups=groups)])

        assert _options(fake_ray_cluster)[0]["concurrency_groups"] == groups

    async def test_absent_concurrency_groups_are_not_passed_to_ray(self, fake_ray_cluster: FakeRayCluster):
        """Passing an empty group mapping would change how ray schedules the actor."""
        await _launch([_make_spec()])

        assert "concurrency_groups" not in _options(fake_ray_cluster)[0]

    async def test_the_cpu_request_comes_from_the_spec(self, fake_ray_cluster: FakeRayCluster):
        """Trainer actors reserve a whole slot, unlike the small command workers."""
        await _launch([_make_spec(num_cpus_per_worker=0.4)])

        assert _options(fake_ray_cluster)[0]["num_cpus"] == 0.4


class TestConcurrencyGroupsAreDeclaredOnce:
    async def test_the_group_an_rpc_method_declares_reaches_ray(self, fake_ray_cluster: FakeRayCluster):
        """A method both wires isolate is declared once, and the launcher is what tells ray about it."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH)])

        assert _GroupedWorker.isolated.__ray_concurrency_group__ == "kill_self"

    async def test_a_group_declared_above_a_wrapper_still_reaches_ray(self, fake_ray_cluster: FakeRayCluster):
        """Ray unwraps a method before reading its group, so a wrapped one would silently run in the default group."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH)])

        assert inspect.unwrap(_GroupedWorker.wrapped_isolated).__ray_concurrency_group__ == "fault_injector"

    async def test_a_default_group_method_is_left_undeclared(self, fake_ray_cluster: FakeRayCluster):
        """Ray rejects an actor naming a group its class never declares, and most methods name none."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH)])

        assert not hasattr(_GroupedWorker.plain, "__ray_concurrency_group__")

    async def test_both_wires_end_up_with_the_same_group(self, fake_ray_cluster: FakeRayCluster):
        """This is the whole point of declaring once: the two wires must not schedule a method differently."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH)])

        specs = collect_rpc_method_specs(_GroupedWorker)
        told_to_ray = {
            name: getattr(inspect.unwrap(getattr(_GroupedWorker, name)), "__ray_concurrency_group__", "default")
            for name in specs
        }

        assert told_to_ray == {name: spec.concurrency_group for name, spec in specs.items()}

    async def test_the_outermost_declaration_is_the_one_ray_hears(self, fake_ray_cluster: FakeRayCluster):
        """Two markers on one method must not resolve differently per wire, whichever one is meant to win."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH)])

        assert inspect.unwrap(_GroupedWorker.outer_declaration_wins).__ray_concurrency_group__ == "kill_self"
        assert collect_rpc_method_specs(_GroupedWorker)["outer_declaration_wins"].concurrency_group == "kill_self"


class TestServeWorkersAreStopped:
    async def test_stopping_kills_the_actor_without_a_graceful_shutdown(self, fake_ray_cluster: FakeRayCluster):
        """Serve workers expose no shutdown rpc, so asking for one only logs noise."""
        manager = await _launch([_make_spec()])

        await manager.stop_cells(["trainer-00000"])

        assert fake_ray_cluster.calls_of("shutdown") == []
        assert fake_ray_cluster.events.count(EVENT_KILL) == 1


class TestServeAndCommandSpecsCoexist:
    async def test_ports_are_allocated_for_serve_cells_too(self, fake_ray_cluster: FakeRayCluster):
        """The trainer master port is allocated by the same path as engine ports."""
        manager = await _launch([_make_spec(num_workers_per_cell=2)])

        addrs = manager.get_addrs()["trainer"]
        assert "master" in addrs[0]
        assert "master" not in addrs[1]


class TestTheBootstrappedClass:
    async def test_evaluates_the_handed_ctor_kwargs_with_the_handed_context(self):
        """The class captures nothing: the recipe and the identity both arrive as constructor arguments."""
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context(worker_in_cell_index=2))

        assert probe.contexts[0].worker_in_cell_index == 2

    async def test_builds_the_context_with_a_backend_capability_of_its_own_process(self, monkeypatch):
        """A spec that asks for its engines is answered by the backend this process sees, not the launcher's."""
        built = _RecordingCapability()
        monkeypatch.setattr(rwm, "_create_ray_backend_capability", lambda: built)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())

        assert probe.contexts[0].capability.cell_operations() is built.operations

    async def test_the_capability_costs_nothing_until_the_spec_asks(self, monkeypatch):
        """Reaching for the worker manager at construction time would make every gpu-less worker pay for it."""
        creations: list[str] = []

        def _create():
            creations.append("created")
            return _RecordingCapability()

        monkeypatch.setattr(rwm, "_create_ray_backend_capability", _create)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())
        assert creations == []

        capability = probe.contexts[0].capability
        capability.cell_operations()
        capability.dynamic_worker_provider(pool_ids=["trainer-engine-actor"])

        assert creations == ["created"]

    async def test_the_capability_forwards_what_the_spec_asked_for(self, monkeypatch):
        """The pool ids a spec names are the ones its provider must watch; dropping them would watch everything."""
        built = _RecordingCapability()
        monkeypatch.setattr(rwm, "_create_ray_backend_capability", lambda: built)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())
        capability = probe.contexts[0].capability
        capability.dynamic_worker_provider(pool_ids=["trainer-engine-actor"])
        capability.static_worker_provider(pool_id="rollout-executor")

        assert built.requested_pool_ids == [["trainer-engine-actor"]]
        assert built.requested_static_pool_ids == ["rollout-executor"]

    async def test_passes_the_computed_keywords_to_the_wrapped_constructor(self):
        """The worker class is keyword-only, exactly as it is when a pod builds it in serve_inner."""
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        worker = actor_class(ctor_kwargs=_CtorKwargsProbe(), context=_launch_context(worker_in_cell_index=2))

        assert worker.kwargs == dict(rank=2, role="actor")

    async def test_keeps_the_name_of_the_class_it_wraps(self):
        """Ray names actors and their errors after the class, and 'BootstrappedWorker' would name them all alike."""
        assert bootstrapped_worker_class(_WORKER_CLASS_PATH).__name__ == DemoServeWorker.__name__
        assert bootstrapped_worker_class(_WORKER_CLASS_PATH).__module__ == DemoServeWorker.__module__

    async def test_survives_cloudpickle_together_with_the_spec_s_ctor_kwargs(self):
        """Ray ships the class and the constructor arguments to the actor process, recipe included."""
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        rebuilt_class, rebuilt_probe = cloudpickle.loads(cloudpickle.dumps((actor_class, _CtorKwargsProbe())))
        worker = rebuilt_class(ctor_kwargs=rebuilt_probe, context=_launch_context(worker_in_cell_index=1))

        assert worker.kwargs == dict(rank=1, role="actor")

    async def test_a_fresh_interpreter_rebuilds_the_class_and_still_evaluates_the_recipe(self):
        """Unpickling in this process hands back the very same class, so only another interpreter rebuilds it."""
        payload = cloudpickle.dumps(
            (
                bootstrapped_worker_class(_WORKER_CLASS_PATH),
                _CtorKwargsProbe(),
                _launch_context(worker_in_cell_index=1),
            )
        )

        completed = subprocess.run(
            [sys.executable, "-c", _REBUILD_IN_CHILD],
            input=payload,
            capture_output=True,
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")])),
            ),
        )

        assert completed.returncode == 0, completed.stderr.decode()
        rebuilt = json.loads(completed.stdout.decode().strip().splitlines()[-1])
        assert rebuilt == dict(rank=1, role="actor", rebuilt=True, name="DemoServeWorker")


class DemoRpcServeWorker:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def echo(self, *, value: int) -> int:
        return value


_RPC_WORKER_CLASS_PATH = f"{DemoRpcServeWorker.__module__}.{DemoRpcServeWorker.__qualname__}"


async def _launch_rpc(spec: ServeWorkerSpec) -> RayWorkerManager:
    return await _launch([spec], comm_backend=WorkerCommBackend.RPC)


class TestServeWorkersUnderRpcComm:
    async def test_the_actor_is_a_serve_actor_rather_than_the_worker_itself(self, fake_ray_cluster: FakeRayCluster):
        """The worker still lives in the actor process, but the actor is the one running the rpc server."""
        await _launch_rpc(_make_spec())

        assert _actor_classes(fake_ray_cluster) == [ServeActor]

    async def test_the_worker_recipe_is_handed_to_the_serve_actor(self, fake_ray_cluster: FakeRayCluster):
        """Evaluating the recipe here would build the worker in the launcher instead of on its gpu."""
        probe = _CtorKwargsProbe()
        await _launch_rpc(_make_spec(ctor_kwargs=probe))

        (ctor_kwargs,) = fake_ray_cluster.ctor_kwargs
        assert list(ctor_kwargs) == ["build_worker"]
        assert probe.contexts == []

    async def test_the_recipe_builds_the_declared_worker_when_the_actor_runs_it(self):
        """What ships must rebuild the very class the spec names, with the rank's own identity."""
        probe = _CtorKwargsProbe()

        worker = cloudpickle.loads(
            cloudpickle.dumps(
                partial(
                    _build_serve_worker,
                    worker_class_path=_WORKER_CLASS_PATH,
                    ctor_kwargs=probe,
                    context=_launch_context(worker_in_cell_index=1),
                )
            )
        )()

        assert isinstance(worker, DemoServeWorker) and worker.kwargs == dict(rank=1, role="actor")

    async def test_the_server_starts_on_the_allocated_rpc_port(self, fake_ray_cluster: FakeRayCluster):
        """The port is only known after allocation, so the server can only be told about it afterwards."""
        manager = await _launch_rpc(_make_spec())

        (call,) = fake_ray_cluster.calls_of("start_rpc_server")
        assert call.kwargs == dict(port=manager.get_addrs()["trainer"][0]["rpc"].port)

    async def test_the_server_starts_after_the_port_is_allocated(self, fake_ray_cluster: FakeRayCluster):
        """Starting first would bind a port nobody advertised, or none at all."""
        await _launch_rpc(_make_spec())

        assert fake_ray_cluster.last_event_index("_get_free_port_block") < fake_ray_cluster.first_event_index(
            "start_rpc_server"
        )

    async def test_every_worker_of_a_cell_serves_its_own_port(self, fake_ray_cluster: FakeRayCluster):
        """Two workers on one node would otherwise be handed the same port and one would fail to bind."""
        await _launch_rpc(_make_spec(num_workers_per_cell=2))

        ports = [call.kwargs["port"] for call in fake_ray_cluster.calls_of("start_rpc_server")]
        assert len(set(ports)) == 2

    async def test_ray_concurrency_groups_are_not_declared(self, fake_ray_cluster: FakeRayCluster):
        """Under rpc the groups are the server's, and ray would schedule methods the actor does not have."""
        await _launch_rpc(_make_spec(concurrency_groups={"heartbeat_status": 1, "default": 1}))

        assert "concurrency_groups" not in _options(fake_ray_cluster)[0]

    async def test_no_server_is_started_under_ray_communication(self, fake_ray_cluster: FakeRayCluster):
        """The two modes coexist, so the ray mode must keep launching the worker as the actor itself."""
        await _launch([_make_spec()])

        assert fake_ray_cluster.calls_of("start_rpc_server") == []


class TestTheHandleAWorkerIsCalledThrough:
    async def test_ray_communication_still_answers_with_a_ray_handle(self, fake_ray_cluster: FakeRayCluster):
        """Nothing about the existing mode changes while both are supported."""
        manager = await _launch([_make_spec()])

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class is None

    async def test_rpc_communication_answers_with_the_class_to_build_a_client_for(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """An rpc handle holds pydantic validators that no cluster can ship, so it is built by its caller."""
        manager = await _launch_rpc(_make_spec())

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class == _WORKER_CLASS_PATH

    async def test_the_provider_turns_that_answer_into_an_rpc_handle(self, fake_ray_cluster: FakeRayCluster):
        """The driver ends up calling the worker over http, which is the whole point of the mode."""
        manager = await _launch_rpc(_make_spec(worker_class=_RPC_WORKER_CLASS_PATH))

        (info,) = manager.get_worker_infos("trainer-00000")
        handle = build_rpc_handle_of_worker_info(info)

        assert isinstance(handle, RpcWorkerHandle)

    async def test_the_handle_points_at_the_address_the_launcher_advertised(self, fake_ray_cluster: FakeRayCluster):
        """A handle aimed anywhere else silently drives another worker on the same node."""
        manager = await _launch_rpc(_make_spec(worker_class=_RPC_WORKER_CLASS_PATH))

        (info,) = manager.get_worker_infos("trainer-00000")
        handle = build_rpc_handle_of_worker_info(info)

        assert handle._transport._server_url == info.self_addrs["rpc"].addr

    async def test_a_worker_that_is_not_served_names_no_class_to_call_it_by(self, fake_ray_cluster: FakeRayCluster):
        """Command workers are called as the actors they are, and an rpc client for them cannot be built."""
        manager = await _launch([_make_spec()])

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class is None
