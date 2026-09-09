from __future__ import annotations

import pytest
from tests.fast.utils.workers.fake_ray import EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

pytestmark = pytest.mark.asyncio


class DemoServeWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def ping(self) -> str:
        return "pong"

    def echo(self, value: str, *, times: int = 1) -> str:
        return value * times


_WORKER_CLASS_PATH = f"{DemoServeWorker.__module__}.{DemoServeWorker.__qualname__}"


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
    await manager.init(specs, pgs if pgs is not None else {})
    return manager


def _actor_classes(cluster: FakeRayCluster) -> list[type]:
    return [handle.actor_class for handle in cluster.handles]


def _options(cluster: FakeRayCluster) -> list[dict]:
    return [handle.options for handle in cluster.handles]


class TestServeWorkersAreLaunched:
    async def test_the_declared_worker_class_is_instantiated(self, fake_ray_cluster: FakeRayCluster):
        """A serve spec names its worker class instead of running a shell command."""
        await _launch([_make_spec(num_workers_per_cell=2)])

        assert _actor_classes(fake_ray_cluster) == [DemoServeWorker, DemoServeWorker]

    async def test_no_launch_command_is_ever_sent(self, fake_ray_cluster: FakeRayCluster):
        """Serve workers start with their constructor, so post_setup must stay silent."""
        await _launch([_make_spec()])

        assert fake_ray_cluster.calls_of("run") == []

    async def test_each_worker_is_constructed_with_its_own_kwargs(self, fake_ray_cluster: FakeRayCluster):
        """Every rank needs its own identity, which is what the launch context carries."""
        await _launch([_make_spec(num_workers_per_cell=3, ctor_kwargs=lambda ctx: {"rank": ctx.worker_in_cell_index})])

        assert [kwargs["rank"] for kwargs in fake_ray_cluster.ctor_kwargs] == [0, 1, 2]

    async def test_gpu_ids_reach_the_constructor(self, fake_ray_cluster: FakeRayCluster):
        """A serve worker cannot ask ray for its slot, so the manager must tell it."""
        spec = _make_spec(
            num_workers_per_cell=2,
            num_gpu_slots_per_worker=1,
            num_gpus_per_worker=0.4,
            pg_name="actor",
            ctor_kwargs=lambda ctx: {"gpu_ids": ctx.gpu_ids},
        )

        await _launch([spec], _make_pgs())

        assert [kwargs["gpu_ids"] for kwargs in fake_ray_cluster.ctor_kwargs] == [[0], [1]]

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
