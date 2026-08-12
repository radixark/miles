from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from ray import cloudpickle
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers import ray_worker_manager as rwm
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager, bootstrapped_worker_class
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec, WorkerLaunchContext

pytestmark = pytest.mark.asyncio


class DemoServeWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def ping(self) -> str:
        return "pong"

    def echo(self, value: str, *, times: int = 1) -> str:
        return value * times


_WORKER_CLASS_PATH = f"{DemoServeWorker.__module__}.{DemoServeWorker.__qualname__}"

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
    method_concurrency_groups: dict[str, str] | None = None,
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
        method_concurrency_groups=method_concurrency_groups,
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


async def _launch(specs, pgs=None) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, pgs if pgs is not None else {})
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
            await manager.init([spec], {})

        assert fake_ray_cluster.handles == []
        assert not manager.get_cell_infos(pool_ids=["trainer"])["trainer-0"].alive


class TestServeSchedulingOptions:
    async def test_concurrency_groups_reach_ray(self, fake_ray_cluster: FakeRayCluster):
        """The trainer heartbeat rpc must not queue behind a running train step."""
        groups = {"heartbeat_status": 1, "default": 1}

        await _launch([_make_spec(concurrency_groups=groups, method_concurrency_groups={"ping": "heartbeat_status"})])

        assert _options(fake_ray_cluster)[0]["concurrency_groups"] == groups

    async def test_absent_concurrency_groups_are_not_passed_to_ray(self, fake_ray_cluster: FakeRayCluster):
        """Passing an empty group mapping would change how ray schedules the actor."""
        await _launch([_make_spec()])

        assert "concurrency_groups" not in _options(fake_ray_cluster)[0]

    async def test_the_routed_methods_are_annotated_on_a_subclass(self, fake_ray_cluster: FakeRayCluster):
        """A declared group nobody is routed to leaves the isolated rpc queued behind the default group."""
        await _launch(
            [
                _make_spec(
                    concurrency_groups={"probe": 1, "default": 1},
                    method_concurrency_groups={"ping": "probe"},
                )
            ]
        )

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class is not DemoServeWorker
        assert actor_class.ping.__ray_concurrency_group__ == "probe"

    async def test_a_worker_without_groups_reaches_ray_unannotated(self, fake_ray_cluster: FakeRayCluster):
        """Ray refuses to build an actor whose method names a group the class never declares."""
        await _launch([_make_spec()])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class is DemoServeWorker
        assert not hasattr(actor_class.ping, "__ray_concurrency_group__")

    async def test_the_cpu_request_comes_from_the_spec(self, fake_ray_cluster: FakeRayCluster):
        """Trainer actors reserve a whole slot, unlike the small command workers."""
        await _launch([_make_spec(num_cpus_per_worker=0.4)])

        assert _options(fake_ray_cluster)[0]["num_cpus"] == 0.4


class TestServeConcurrencyGroupRouting:
    async def test_the_declared_worker_class_stays_unannotated(self, fake_ray_cluster: FakeRayCluster):
        """Annotating the class itself would follow every later non-fault-tolerant run of that class."""
        await _launch(
            [
                _make_spec(
                    concurrency_groups={"probe": 1, "default": 1},
                    method_concurrency_groups={"ping": "probe"},
                )
            ]
        )

        assert not hasattr(DemoServeWorker.ping, "__ray_concurrency_group__")

    async def test_each_routed_method_lands_in_its_own_group(self, fake_ray_cluster: FakeRayCluster):
        """Collapsing every routed method into one group serializes the heartbeat with the fault injector."""
        await _launch(
            [
                _make_spec(
                    concurrency_groups={"probe": 1, "killer": 1, "default": 1},
                    method_concurrency_groups={"ping": "probe", "echo": "killer"},
                )
            ]
        )

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert (actor_class.ping.__ray_concurrency_group__, actor_class.echo.__ray_concurrency_group__) == (
            "probe",
            "killer",
        )

    async def test_an_unrouted_method_is_inherited_untouched(self, fake_ray_cluster: FakeRayCluster):
        """A train step pushed out of the default group would no longer block the group it must own."""
        await _launch(
            [
                _make_spec(
                    concurrency_groups={"probe": 1, "default": 1},
                    method_concurrency_groups={"ping": "probe"},
                )
            ]
        )

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class.echo is DemoServeWorker.echo

    async def test_a_routed_method_still_runs_the_original_body(self, fake_ray_cluster: FakeRayCluster):
        """A wrapper that swallowed the arguments or the return value would break every isolated rpc."""
        await _launch(
            [
                _make_spec(
                    concurrency_groups={"probe": 1, "default": 1},
                    method_concurrency_groups={"echo": "probe"},
                )
            ]
        )

        worker = _actor_classes(fake_ray_cluster)[0]()
        assert worker.echo("ab", times=2) == "abab"


class TestServeWorkersAreStopped:
    async def test_stopping_kills_the_actor_without_a_graceful_shutdown(self, fake_ray_cluster: FakeRayCluster):
        """Serve workers expose no shutdown rpc, so asking for one only logs noise."""
        manager = await _launch([_make_spec()])

        await manager.stop_cells(["trainer-0"])

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
