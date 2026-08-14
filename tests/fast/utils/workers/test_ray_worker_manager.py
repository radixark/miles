from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from tests.fast.utils.workers.fake_ray import EVENT_CREATE, EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers import ray_worker_manager
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, PortInfo, SchedulingSpec


@dataclass
class _LaunchRecorder:
    contexts: list[LaunchCommandContext] = field(default_factory=list)

    def command(self, ctx: LaunchCommandContext) -> str:
        self.contexts.append(ctx)
        return f"run-{ctx.cell_index}-{ctx.worker_in_cell_index}"

    def context_of(self, *, cell_index: int, worker_in_cell_index: int) -> LaunchCommandContext:
        matches = [
            ctx
            for ctx in self.contexts
            if ctx.cell_index == cell_index and ctx.worker_in_cell_index == worker_in_cell_index
        ]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]


def _make_spec(
    name: str,
    *,
    num_cells: int = 1,
    num_workers_per_cell: int = 1,
    port_infos: list[PortInfo] | None = None,
    env_var: dict[str, str] | None = None,
    launch_command: Callable[[LaunchCommandContext], str] | None = None,
    num_gpus_per_worker: float = 0,
    num_gpu_slots_per_worker: int = 0,
    pg_name: str | None = None,
    pg_slot_offset: int = 0,
    pin_to_head: bool = False,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        port_infos=(
            port_infos if port_infos is not None else [PortInfo(name="primary", static_port=8000, allow_dynamic=True)]
        ),
        env_var=lambda _ctx: dict(env_var or {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
            num_gpu_slots_per_worker=num_gpu_slots_per_worker,
            pg_name=pg_name,
            pg_slot_offset=pg_slot_offset,
            pin_to_head=pin_to_head,
        ),
        launch_command=launch_command if launch_command is not None else (lambda ctx: "sleep 600"),
    )


def _make_pgs(*, num_slots: int = 8, first_gpu_id: int = 0) -> dict[str, PlacementGroupInfo]:
    return {
        "rollout": PlacementGroupInfo(
            pg="fake-pg",
            pg_reordered_bundle_indices=[(slot * 3 + 1) % num_slots for slot in range(num_slots)],
            pg_reordered_gpu_ids=[first_gpu_id + slot for slot in range(num_slots)],
        )
    }


async def _launch(
    specs: list[CommandWorkerSpec], pgs: dict[str, PlacementGroupInfo] | None = None
) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(specs, pgs if pgs is not None else {})
    return manager


class TestLaunchEntryPoint:
    async def test_the_manager_is_registered_under_its_well_known_name(self, fake_ray_cluster: FakeRayCluster):
        """Consumers find the manager by a fixed actor name, so it must be launched under that name."""
        handle = RayWorkerManager.launch([], {})

        assert handle.options["name"] == "ray_worker_manager"
        assert handle.actor_class is RayWorkerManager
        assert [call.method for call in fake_ray_cluster.calls] == ["init"]

    async def test_launch_waits_for_init_to_finish(self, fake_ray_cluster: FakeRayCluster):
        """Returning before init completes would expose a manager whose workers have no addresses yet."""
        specs = [_make_spec("router")]
        pgs: dict = {}

        RayWorkerManager.launch(specs, pgs)

        init_calls = fake_ray_cluster.calls_of("init")
        assert len(init_calls) == 1
        assert init_calls[0].args == (specs, pgs)
        assert fake_ray_cluster.resolved_refs == ["init"]

    async def test_launch_propagates_an_init_failure(self, fake_ray_cluster: FakeRayCluster):
        """A pool that failed to come up must not look like a successful launch."""
        fake_ray_cluster.method_errors["init"] = RuntimeError("init exploded")

        with pytest.raises(RuntimeError, match="init exploded"):
            RayWorkerManager.launch([_make_spec("router")], {})

    async def test_get_handle_resolves_the_same_well_known_name(self, fake_ray_cluster: FakeRayCluster):
        """The lookup helper and the launcher must agree on the actor name."""
        RayWorkerManager.launch([], {})

        assert RayWorkerManager.get_handle() is fake_ray_cluster.named_actors["ray_worker_manager"]


class TestInitLaunchesWorkers:
    async def test_creates_one_command_actor_per_worker_of_every_cell(self, fake_ray_cluster: FakeRayCluster):
        """Every cell of every spec gets its own actor per worker slot."""
        await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2), _make_spec("router")])

        assert len(fake_ray_cluster.handles) == 5
        assert {handle.actor_class for handle in fake_ray_cluster.handles} == {CommandActor}
        assert fake_ray_cluster.ctor_kwargs == [{} for _ in range(5)]

    async def test_a_spec_without_cells_launches_nothing(self, fake_ray_cluster: FakeRayCluster):
        """A disabled spec contributes no workers instead of an idle one."""
        await _launch([_make_spec("session-server", num_cells=0)])

        assert fake_ray_cluster.handles == []

    async def test_duplicate_pool_names_are_rejected(self, fake_ray_cluster: FakeRayCluster):
        """Two specs sharing a name would collide in the worker registry, so init fails fast."""
        with pytest.raises(AssertionError):
            await _launch([_make_spec("router"), _make_spec("router")])

    async def test_the_specs_env_vars_become_the_actors_runtime_env(self, fake_ray_cluster: FakeRayCluster):
        """The spec's env vars are handed to ray as the actor's runtime env."""
        await _launch([_make_spec("router", env_var={"MILES_TEST_VAR": "7"})])

        assert fake_ray_cluster.handles[0].options["runtime_env"] == {"env_vars": {"MILES_TEST_VAR": "7"}}

    async def test_each_phase_completes_for_all_workers_before_the_next_starts(self, fake_ray_cluster: FakeRayCluster):
        """Launching, port allocation and command start are global barriers, so every worker sees complete state."""
        await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2), _make_spec("router")])

        assert fake_ray_cluster.last_event_index(EVENT_CREATE) < fake_ray_cluster.first_event_index("_get_node_ip")
        assert fake_ray_cluster.last_event_index("_get_free_port_block") < fake_ray_cluster.first_event_index("run")

    async def test_a_failing_phase_stops_the_pipeline(self, fake_ray_cluster: FakeRayCluster):
        """A worker that cannot allocate its ports must not leave other workers starting their commands."""
        spec = _make_spec("engine", num_workers_per_cell=2)

        async def failing_alloc(self) -> None:
            raise RuntimeError("no ports")

        manager = RayWorkerManager()
        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(
                    "miles.utils.workers.ray_worker_manager._CommandActorManager.alloc_ports", failing_alloc
                )
                await manager.init([spec], {})

        assert len(fake_ray_cluster.handles) == 2
        assert fake_ray_cluster.calls_of("run") == []


class TestInitAllocatesPorts:
    async def test_dynamic_ports_of_one_node_never_overlap(self, fake_ray_cluster: FakeRayCluster):
        """Workers sharing a node must be handed distinct ports."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])

        ports = [
            manager.get_worker_addrs(f"engine-{cell_index}-{worker_in_cell_index}")["primary"].port
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ]
        assert len(set(ports)) == 4

    async def test_a_consecutive_port_block_is_reserved_as_a_whole(self, fake_ray_cluster: FakeRayCluster):
        """A worker asking for a port block leaves the whole block out of the next worker's reach."""
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True, num_consecutive=5)],
        )
        manager = await _launch([spec])

        first = manager.get_worker_addrs("engine-0-0")["primary"].port
        second = manager.get_worker_addrs("engine-0-1")["primary"].port
        assert second >= first + 5
        assert [call.kwargs["count"] for call in fake_ray_cluster.calls_of("_get_free_port_block")] == [5, 5]

    async def test_static_ports_bypass_the_allocator(self, fake_ray_cluster: FakeRayCluster):
        """A port the worker cannot choose is taken from the spec verbatim, without probing the node."""
        spec = _make_spec(
            "router",
            port_infos=[
                PortInfo(name="primary", static_port=7777, allow_dynamic=False),
                PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
            ],
        )
        manager = await _launch([spec])

        assert manager.get_worker_addrs("router-0-0")["primary"].port == 7777
        assert len(fake_ray_cluster.calls_of("_get_free_port_block")) == 1

    async def test_ports_are_tracked_per_node(self, fake_ray_cluster: FakeRayCluster):
        """Workers on different nodes may reuse the same port number."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        first = manager.get_worker_addrs("engine-0-0")["primary"]
        second = manager.get_worker_addrs("engine-0-1")["primary"]
        assert (first.host, second.host) == ("10.0.0.1", "10.0.0.2")
        assert first.port == second.port

    async def test_the_worker_addr_host_is_the_node_the_actor_landed_on(self, fake_ray_cluster: FakeRayCluster):
        """Addresses advertise the actor's own node, not the driver's."""
        fake_ray_cluster.use_node_ips("10.1.2.3")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addrs("router-0-0")["primary"].host == "10.1.2.3"

    async def test_ipv6_hosts_are_bracketed(self, fake_ray_cluster: FakeRayCluster):
        """An ipv6 node address is advertised in url-safe bracketed form."""
        fake_ray_cluster.use_node_ips("2001:db8::7")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addrs("router-0-0")["primary"].host == "[2001:db8::7]"


class TestPortAllocationDetails:
    async def test_the_allocator_probes_the_workers_own_actor_on_its_own_node(self, fake_ray_cluster: FakeRayCluster):
        """Ports must be probed on the node that will bind them, through that worker's own actor."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        await _launch([_make_spec("engine", num_workers_per_cell=2)])

        probes = fake_ray_cluster.calls_of("_get_free_port_block")
        assert [probe.handle for probe in probes] == fake_ray_cluster.handles
        assert [probe.handle.node_ip for probe in probes] == ["10.0.0.1", "10.0.0.2"]

    async def test_every_declared_port_is_allocated_in_spec_order(self, fake_ray_cluster: FakeRayCluster):
        """A worker is launched with an address for every port name its spec declares."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            launch_command=recorder.command,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
                PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            ],
        )
        await _launch([spec])

        addrs = recorder.context_of(cell_index=0, worker_in_cell_index=0).self_addrs
        assert list(addrs) == ["primary", "nccl", "engine_info_bootstrap"]
        assert len({addr.port for addr in addrs.values()}) == 3

    async def test_workers_of_different_specs_never_share_a_port(self, fake_ray_cluster: FakeRayCluster):
        """One allocator serves the whole pool, so specs cannot hand out the same port twice."""
        manager = await _launch([_make_spec("router", num_workers_per_cell=2), _make_spec("engine", num_cells=2)])

        ports = [
            manager.get_worker_addrs(name)["primary"].port
            for name in ["router-0-0", "router-0-1", "engine-0-0", "engine-1-0"]
        ]
        assert len(set(ports)) == 4


class TestInitStartsCommands:
    async def test_every_worker_runs_the_command_rendered_for_it(self, fake_ray_cluster: FakeRayCluster):
        """Each worker's actor runs exactly the command its own launch context rendered."""
        recorder = _LaunchRecorder()
        await _launch([_make_spec("engine", num_cells=2, launch_command=recorder.command)])

        run_calls = fake_ray_cluster.calls_of("run")
        assert [call.kwargs["cmd"] for call in run_calls] == ["run-0-0", "run-1-0"]
        assert [call.kwargs["envs"] for call in run_calls] == [{}, {}]

    async def test_the_launch_context_carries_the_workers_own_indices_and_addrs(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """A worker's launch context describes that worker: its cell, its slot and its own addresses."""
        recorder = _LaunchRecorder()
        spec = _make_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=recorder.command)
        manager = await _launch([spec])

        assert {(ctx.cell_index, ctx.worker_in_cell_index) for ctx in recorder.contexts} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        for cell_index in range(2):
            for worker_in_cell_index in range(2):
                ctx = recorder.context_of(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
                expected = manager.get_worker_addrs(f"engine-{cell_index}-{worker_in_cell_index}")["primary"]
                assert ctx.self_addrs["primary"] == expected


class TestSpecEnvVars:
    async def test_each_spec_contributes_its_own_env_to_its_own_workers(self, fake_ray_cluster: FakeRayCluster):
        """Env vars are per spec, so one spec's variables must not leak into another spec's workers."""
        await _launch(
            [
                _make_spec("router", env_var={"ROUTER_ONLY": "1"}),
                _make_spec("engine", num_cells=2, env_var={"ENGINE_ONLY": "2"}),
            ]
        )

        assert [handle.options["runtime_env"] for handle in fake_ray_cluster.handles] == [
            {"env_vars": {"ROUTER_ONLY": "1"}},
            {"env_vars": {"ENGINE_ONLY": "2"}},
            {"env_vars": {"ENGINE_ONLY": "2"}},
        ]

    async def test_the_env_of_a_spec_is_resolved_once_per_worker(self, fake_ray_cluster: FakeRayCluster):
        """The spec's env callable is what the manager stores, evaluated per worker rather than cached globally."""
        calls: list[int] = []

        def _env(_ctx) -> dict[str, str]:
            calls.append(len(calls))
            return {"CALL_INDEX": str(len(calls))}

        spec = CommandWorkerSpec(
            name="engine",
            port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True)],
            env_var=_env,
            scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=1, num_gpus_per_worker=0),
            launch_command=lambda ctx: "sleep 600",
        )
        await _launch([spec])

        assert len(calls) == 2
        assert [handle.options["runtime_env"]["env_vars"]["CALL_INDEX"] for handle in fake_ray_cluster.handles] == [
            "1",
            "2",
        ]


class TestActorResources:
    async def test_every_worker_reserves_a_fraction_of_a_cpu(self, fake_ray_cluster: FakeRayCluster):
        """Workers are launchers, not compute, so they must not each claim a whole cpu."""
        await _launch([_make_spec("router")])

        assert fake_ray_cluster.handles[0].options["num_cpus"] == 0.2


class TestFailureModes:
    async def test_a_failed_worker_launch_stops_before_any_port_is_allocated(self, fake_ray_cluster: FakeRayCluster):
        """Nothing downstream may run when the pool could not even be created."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        async def failing_launch(self) -> None:
            raise RuntimeError("no capacity")

        with pytest.raises(RuntimeError, match="no capacity"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(_CommandActorManager, "launch_actor", failing_launch)
                await _launch([_make_spec("engine", num_cells=2)])

        assert fake_ray_cluster.calls_of("_get_free_port_block") == []
        assert fake_ray_cluster.calls_of("run") == []

    async def test_a_command_that_cannot_be_rendered_starts_nothing(self, fake_ray_cluster: FakeRayCluster):
        """A spec whose launch command raises must not leave half of the pool running."""

        def _explode(ctx: LaunchCommandContext) -> str:
            raise ValueError("cannot render")

        with pytest.raises(ValueError, match="cannot render"):
            await _launch([_make_spec("engine", num_cells=2, launch_command=_explode)])

        assert fake_ray_cluster.calls_of("run") == []


class TestGetWorkerAddrs:
    async def test_names_are_the_pool_with_the_cell_and_worker_index(self, fake_ray_cluster: FakeRayCluster):
        """Workers are addressable under ``<spec>-<cell>-<worker>``."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])

        for name in ["engine-0-0", "engine-0-1", "engine-1-0", "engine-1-1"]:
            assert manager.get_worker_addrs(name)["primary"].port > 0

    async def test_an_unknown_worker_name_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        """Looking up a worker that does not exist must not silently return an arbitrary address."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        with pytest.raises(AssertionError):
            manager.get_worker_addrs("engine-2-0")["primary"]


class TestPinToHead:
    async def test_a_pinned_worker_keeps_its_resources_and_gains_head_affinity(
        self, fake_ray_cluster: FakeRayCluster, monkeypatch: pytest.MonkeyPatch
    ):
        """Pinning adds the head-affinity strategy without dropping the other actor options."""
        monkeypatch.setattr(
            "miles.utils.workers.ray_worker_manager.compute_ray_pin_head_options",
            lambda: {"scheduling_strategy": "head-affinity"},
        )
        await _launch([_make_spec("router", pin_to_head=True, env_var={"A": "1"})])

        options = fake_ray_cluster.handles[0].options
        assert options["scheduling_strategy"] == "head-affinity"
        assert options["num_cpus"] == 0.2
        assert options["runtime_env"] == {"env_vars": {"A": "1"}}

    async def test_an_unpinned_worker_never_resolves_the_head_node(
        self, fake_ray_cluster: FakeRayCluster, monkeypatch: pytest.MonkeyPatch
    ):
        """Unpinned workers must not even look the head node up, let alone all land on it."""

        def _fail() -> dict:
            raise AssertionError("the head node must not be resolved for unpinned workers")

        monkeypatch.setattr("miles.utils.workers.ray_worker_manager.compute_ray_pin_head_options", _fail)
        await _launch([_make_spec("router")])

        assert "scheduling_strategy" not in fake_ray_cluster.handles[0].options


class TestSpecAddrs:
    async def test_every_worker_sees_the_addresses_of_all_specs(self, fake_ray_cluster: FakeRayCluster):
        """Cross-spec wiring works because each command is rendered with the whole pool's addresses."""
        recorder = _LaunchRecorder()
        manager = await _launch(
            [
                _make_spec("inference-router-0"),
                _make_spec("session-server", num_cells=2, launch_command=recorder.command),
            ]
        )

        for ctx in recorder.contexts:
            assert sorted(ctx.spec_addrs) == ["inference-router-0", "session-server"]
            assert (
                ctx.spec_addrs["inference-router-0"][0]["primary"]
                == manager.get_worker_addrs("inference-router-0-0-0")["primary"]
            )
            assert [addr["primary"] for addr in ctx.spec_addrs["session-server"]] == [
                manager.get_worker_addrs("session-server-0-0")["primary"],
                manager.get_worker_addrs("session-server-1-0")["primary"],
            ]


class TestGetAddrs:
    async def test_groups_are_listed_in_cell_then_worker_order(self, fake_ray_cluster: FakeRayCluster):
        """Consumers index this map positionally, so the order must follow cells and then workers."""
        recorder = _LaunchRecorder()
        manager = await _launch(
            [_make_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=recorder.command)]
        )

        expected = [
            manager.get_worker_addrs(f"engine-{cell_index}-{worker_in_cell_index}")["primary"]
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ]
        assert [addr["primary"] for addr in manager.get_addrs()["engine"]] == expected

    async def test_a_disabled_group_is_listed_as_empty(self, fake_ray_cluster: FakeRayCluster):
        """A spec with no cells is still visible to consumers, with no addresses."""
        manager = await _launch([_make_spec("router"), _make_spec("session-server", num_cells=0)])

        assert manager.get_addrs()["session-server"] == []
        assert len(manager.get_addrs()["router"]) == 1


class TestMasterPorts:
    async def test_a_master_port_is_allocated_once_per_cell(self, fake_ray_cluster: FakeRayCluster):
        """Only worker 0 reserves the cell's master port, so peers cannot each take their own."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=3,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True, num_consecutive=5),
            ],
        )
        manager = await _launch([spec])

        assert [call.kwargs["count"] for call in fake_ray_cluster.calls_of("_get_free_port_block")].count(5) == 2
        addrs = manager.get_addrs()["engine"]
        assert [sorted(addr) for addr in addrs] == [
            ["dist_init", "primary"],
            ["primary"],
            ["primary"],
            ["dist_init", "primary"],
            ["primary"],
            ["primary"],
        ]

    async def test_a_static_master_port_is_recorded_only_on_worker_zero(self, fake_ray_cluster: FakeRayCluster):
        """A pinned master port is not allocated, and still belongs to worker 0 alone."""
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9123, mode="master", allow_dynamic=False),
            ],
        )
        manager = await _launch([spec])

        addrs = manager.get_addrs()["engine"]
        assert addrs[0]["dist_init"].port == 9123
        assert "dist_init" not in addrs[1]
        assert len(fake_ray_cluster.calls_of("_get_free_port_block")) == 2

    async def test_every_worker_of_a_cell_launches_with_that_cells_master_addr(self, fake_ray_cluster: FakeRayCluster):
        """All ranks of a cell must be told the same master endpoint, and never another cell's."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        await _launch([spec])

        masters = {
            (cell_index, worker_in_cell_index): recorder.context_of(
                cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
            ).self_addrs["dist_init"]
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        }
        assert masters[(0, 0)] == masters[(0, 1)]
        assert masters[(1, 0)] == masters[(1, 1)]
        assert masters[(0, 0)] != masters[(1, 0)]

    async def test_a_master_addr_keeps_rank_zeros_host_on_another_node(self, fake_ray_cluster: FakeRayCluster):
        """A peer on a second node must dial rank 0's host, not its own, or torch.distributed hangs."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        await _launch([spec])

        peer = recorder.context_of(cell_index=0, worker_in_cell_index=1).self_addrs
        assert peer["primary"].host == "10.0.0.2"
        assert peer["dist_init"].host == "10.0.0.1"

    async def test_a_master_addr_does_not_overwrite_a_workers_own_ports(self, fake_ray_cluster: FakeRayCluster):
        """Merging the cell's master addr must leave each worker's own addresses intact."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        manager = await _launch([spec])

        for worker_in_cell_index in range(2):
            ctx = recorder.context_of(cell_index=0, worker_in_cell_index=worker_in_cell_index)
            assert ctx.self_addrs["primary"] == manager.get_worker_addrs(f"engine-0-{worker_in_cell_index}")["primary"]


class TestConcurrentPhases:
    async def test_all_cells_of_a_phase_run_concurrently(self, fake_ray_cluster: FakeRayCluster):
        """Cells are configured concurrently, so a large pool does not start up one cell at a time."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        original_alloc_ports = _CommandActorManager.alloc_ports
        entered: list[int] = []
        release = asyncio.Event()

        async def gated_alloc(self) -> None:
            entered.append(self.parent.cell_index)
            if len(entered) == 3:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=5)
            await original_alloc_ports(self)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc)
            manager = await asyncio.wait_for(_launch([_make_spec("engine", num_cells=3)]), timeout=10)

        assert sorted(entered) == [0, 1, 2]
        assert len({manager.get_worker_addrs(f"engine-{index}-0")["primary"].port for index in range(3)}) == 3


class TestGpuPlacement:
    async def test_gpu_slots_expand_over_cells_and_workers(self, fake_ray_cluster: FakeRayCluster):
        """Each worker occupies its own contiguous slot span after its group's base offset."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
            pg_slot_offset=2,
            launch_command=recorder.command,
        )
        await _launch([spec], _make_pgs(num_slots=10, first_gpu_id=0))

        assert [
            recorder.context_of(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index).gpu_ids
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ] == [[2, 3], [4, 5], [6, 7], [8, 9]]

    async def test_gpu_ids_come_from_the_reordered_slot_mapping(self, fake_ray_cluster: FakeRayCluster):
        """A worker's gpu ids start at the node-local id its slot maps to, not at the slot index."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
            launch_command=recorder.command,
        )
        await _launch([spec], _make_pgs(num_slots=8, first_gpu_id=4))

        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).gpu_ids == [4, 5]
        assert recorder.context_of(cell_index=0, worker_in_cell_index=1).gpu_ids == [6, 7]

    async def test_a_pg_worker_is_scheduled_on_the_bundle_its_slot_maps_to(self, fake_ray_cluster: FakeRayCluster):
        """The actor is placed on the reordered bundle of its slot, not on the raw slot index."""
        pgs = _make_pgs(num_slots=8)
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=1,
            pg_name="rollout",
            pg_slot_offset=3,
        )
        await _launch([spec], pgs)

        strategies = [handle.options["scheduling_strategy"] for handle in fake_ray_cluster.handles]
        assert [strategy.placement_group_bundle_index for strategy in strategies] == [
            pgs["rollout"].pg_reordered_bundle_indices[3],
            pgs["rollout"].pg_reordered_bundle_indices[4],
        ]
        assert all(strategy.placement_group == "fake-pg" for strategy in strategies)
        assert all(handle.options["num_gpus"] == 0.2 for handle in fake_ray_cluster.handles)

    async def test_a_worker_without_a_pg_asks_for_no_placement(self, fake_ray_cluster: FakeRayCluster):
        """Specs that do not own gpus are scheduled anywhere and get no gpu ids."""
        recorder = _LaunchRecorder()
        await _launch([_make_spec("router", launch_command=recorder.command)])

        assert "scheduling_strategy" not in fake_ray_cluster.handles[0].options
        assert fake_ray_cluster.handles[0].options["num_gpus"] == 0
        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).gpu_ids == []


class TestPgActorResources:
    async def test_the_gpu_request_comes_from_the_spec(self, fake_ray_cluster: FakeRayCluster):
        """A spec that declares a gpu fraction gets exactly that, so co-located engines still fit."""
        await _launch([_make_spec("engine", num_gpus_per_worker=0.2)])

        assert fake_ray_cluster.handles[0].options["num_gpus"] == 0.2

    async def test_pg_workers_capture_their_child_tasks_in_the_group(self, fake_ray_cluster: FakeRayCluster):
        """Child tasks must stay inside the placement group, or they escape the reserved gpus."""
        spec = _make_spec(
            "engine",
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=1,
            pg_name="rollout",
        )
        await _launch([spec], _make_pgs())

        assert fake_ray_cluster.handles[0].options["scheduling_strategy"].placement_group_capture_child_tasks is True


class TestPgFailureModes:
    async def test_a_spec_pointing_at_an_unknown_placement_group_fails(self, fake_ray_cluster: FakeRayCluster):
        """Mistyping the placement group name must fail instead of silently ignoring placement."""
        spec = _make_spec("engine", num_gpus_per_worker=0.2, num_gpu_slots_per_worker=1, pg_name="typo")

        with pytest.raises(KeyError):
            await _launch([spec], _make_pgs())

        assert fake_ray_cluster.calls_of("run") == []


class TestCellLifecycle:
    async def test_a_cell_launches_its_workers_only_once(self, fake_ray_cluster: FakeRayCluster):
        """Launching an already-populated cell again would orphan the first generation of actors."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]

        with pytest.raises(AssertionError):
            await cell.launch_actors()

        assert len(fake_ray_cluster.handles) == 2

    async def test_the_stages_of_a_cell_fan_out_to_every_worker(self, fake_ray_cluster: FakeRayCluster):
        """Each stage covers all workers of the cell, so no rank is left half-configured."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=3)])

        assert len(fake_ray_cluster.calls_of("_get_node_ip")) == 3
        assert len(fake_ray_cluster.calls_of("run")) == 3
        assert len(manager._pools["engine"].cells[0].actors) == 3


class TestCellStop:
    async def test_stopping_a_cell_shuts_down_and_kills_every_worker(self, fake_ray_cluster: FakeRayCluster):
        """A stopped cell releases all of its workers, not just the first one."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]

        await cell.stop()

        assert [call.method for call in fake_ray_cluster.calls if call.method == "shutdown"] == ["shutdown"] * 2
        assert fake_ray_cluster.events.count(EVENT_KILL) == 2
        assert all(handle.killed for handle in fake_ray_cluster.handles)
        assert cell.actors is None

    async def test_stopping_one_cell_leaves_the_others_running(self, fake_ray_cluster: FakeRayCluster):
        """Stopping a cell must not touch the workers of its siblings."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager._pools["engine"].cells[0].stop()

        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, False]
        assert manager._pools["engine"].cells[1].actors is not None

    async def test_a_worker_that_cannot_shut_down_gracefully_is_still_killed(self, fake_ray_cluster: FakeRayCluster):
        """A hung or broken worker is exactly the one that must be force-killed."""
        manager = await _launch([_make_spec("engine")])
        cell = manager._pools["engine"].cells[0]
        fake_ray_cluster.handles[0].failing_methods["shutdown"] = RuntimeError("shutdown timed out")

        await cell.stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1
        assert cell.actors is None

    async def test_a_failing_kill_does_not_abort_the_teardown_of_other_workers(self, fake_ray_cluster: FakeRayCluster):
        """One worker ray cannot kill must not stop the cell from tearing the rest down."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]
        fake_ray_cluster.kill_error = RuntimeError("kill failed")

        await cell.stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 2
        assert cell.actors is None


class TestStopDetails:
    async def test_each_worker_is_asked_to_shut_down_before_it_is_killed(self, fake_ray_cluster: FakeRayCluster):
        """Graceful shutdown first gives the subprocess a chance to exit cleanly."""
        manager = await _launch([_make_spec("engine")])

        await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events[-2:] == ["shutdown", EVENT_KILL]

    async def test_graceful_shutdown_is_bounded_by_a_timeout(self, fake_ray_cluster: FakeRayCluster):
        """A hung worker must not block teardown forever."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].hanging_methods["shutdown"] = 3600

        with patch.object(ray_worker_manager, "_SHUTDOWN_TIMEOUT", 0.01):
            await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_shutdown_leaves_the_managers_event_loop_free(self, fake_ray_cluster: FakeRayCluster):
        """A slow shutdown must not block the manager loop, which still has to answer cell polls meanwhile."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].hanging_methods["shutdown"] = 0.05
        polls: int = 0

        async def poll_cells() -> None:
            nonlocal polls
            while True:
                assert "engine-0" in manager.get_cell_infos(pool_ids=["engine"])
                polls += 1
                await asyncio.sleep(0)

        stop_task = asyncio.create_task(manager._pools["engine"].cells[0].stop())
        poll_task = asyncio.create_task(poll_cells())
        await stop_task
        polls_during_stop: int = polls
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)

        assert polls_during_stop > 20

    async def test_all_workers_are_killed_even_when_one_shutdown_hangs(self, fake_ray_cluster: FakeRayCluster):
        """One broken worker must not save its peers from teardown."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=3)])
        fake_ray_cluster.handles[1].failing_methods["shutdown"] = TimeoutError("hung")

        await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 3
        assert all(handle.killed for handle in fake_ray_cluster.handles)


class TestGetWorkerInfos:
    async def test_describes_every_worker_of_only_the_requested_cell(self, fake_ray_cluster: FakeRayCluster):
        """A consumer asking about one cell gets that cell's workers, in rank order, fully described."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
        )
        manager = await _launch([spec], _make_pgs(num_slots=8))

        infos = manager.get_worker_infos("engine-1")

        assert [info.name for info in infos] == ["engine-1-0", "engine-1-1"]
        assert [info.generation for info in infos] == [1, 1]
        assert [info.gpu_ids for info in infos] == [[4, 5], [6, 7]]
        assert [info.self_addrs for info in infos] == manager.get_addrs()["engine"][2:]
        assert [info.actor_handle for info in infos] == fake_ray_cluster.handles[2:]

    async def test_a_stopped_cell_reports_no_workers_instead_of_raising(self, fake_ray_cluster: FakeRayCluster):
        """A cell stopped between snapshot and round-trip must report an empty worker list, not crash."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        infos = manager.get_worker_infos("engine-0")

        assert infos == []


class TestGetWorkerInfosErrors:
    async def test_an_unknown_pool_is_reported_instead_of_guessed(self, fake_ray_cluster: FakeRayCluster):
        """Asking about a spec the manager never launched must fail loudly."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.get_worker_infos("router-0")

    async def test_a_cell_index_beyond_the_group_is_reported(self, fake_ray_cluster: FakeRayCluster):
        """Asking about a cell that does not exist must fail rather than return another cell."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        with pytest.raises(AssertionError):
            manager.get_worker_infos("engine-2")


class TestGetCellInfos:
    async def test_only_the_asked_for_specs_are_reported(self, fake_ray_cluster: FakeRayCluster):
        """The controller owns engines only; reconciling a router would try to serve from it."""
        manager = await _launch([_make_spec("engine", num_cells=2), _make_spec("router")])

        infos = manager.get_cell_infos(pool_ids=["engine"])

        assert sorted(infos) == ["engine-0", "engine-1"]

    async def test_every_info_carries_the_spec_it_came_from(self, fake_ray_cluster: FakeRayCluster):
        """The spec name is what makes a cell's role explicit instead of sniffed from its meta."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].pool_id == "engine"

    async def test_an_unknown_pool_is_reported_instead_of_silently_empty(self, fake_ray_cluster: FakeRayCluster):
        """A renamed spec would otherwise make every consumer see zero cells and blame something else."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.get_cell_infos(pool_ids=["engine", "router"])

    async def test_asking_for_nothing_reports_nothing(self, fake_ray_cluster: FakeRayCluster):
        """A run with no engines of its own must not fall back to seeing everyone else's."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=[]) == {}

    async def test_every_info_says_whether_its_cell_still_has_a_process(self, fake_ray_cluster: FakeRayCluster):
        """A suspended cell must stay listed, or nothing could ever ask for it to be resumed."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive


class TestStartAndStopCells:
    async def test_stopping_by_id_releases_only_the_named_cell(self, fake_ray_cluster: FakeRayCluster):
        """Suspending one engine must leave its siblings serving."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-0"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-0"].alive
        assert infos["engine-1"].alive
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, False]

    async def test_a_stopped_cell_comes_back_on_start(self, fake_ray_cluster: FakeRayCluster):
        """Resume is a fresh launch of the same cell, so the cell reappears to the reconciler."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-0"])

        await manager.start_cells(["engine-0"])

        assert all(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_restarted_cell_reports_a_new_workers_hash(self, fake_ray_cluster: FakeRayCluster):
        """The hash is what tells the trainer to rebuild its connection to the new process."""
        manager = await _launch([_make_spec("engine")])
        before = manager.get_cell_infos(pool_ids=["engine"])["engine-0"].workers_hash

        await manager.stop_cells(["engine-0"])
        await manager.start_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].workers_hash != before

    async def test_a_restarted_cell_runs_its_command_again(self, fake_ray_cluster: FakeRayCluster):
        """Resume must relaunch the subprocess, not merely re-register the old actors."""
        manager = await _launch([_make_spec("engine")])

        await manager.stop_cells(["engine-0"])
        await manager.start_cells(["engine-0"])

        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_stopping_an_already_stopped_cell_is_a_noop(self, fake_ray_cluster: FakeRayCluster):
        """Heal loops retry, so a redundant suspend must not blow up on missing actors."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        await manager.stop_cells(["engine-0"])

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_starting_an_already_running_cell_is_a_noop(self, fake_ray_cluster: FakeRayCluster):
        """A repeated resume must not relaunch a live cell, nor fail the request that asked for it."""
        manager = await _launch([_make_spec("engine")])
        created_before = fake_ray_cluster.events.count(EVENT_CREATE)

        await manager.start_cells(["engine-0"])

        assert fake_ray_cluster.events.count(EVENT_CREATE) == created_before
        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive

    async def test_a_cell_whose_actor_is_already_dead_still_stops(self, fake_ray_cluster: FakeRayCluster):
        """Suspend is step one of healing a crashed cell, so a dead actor is the normal case."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].failing_methods["shutdown"] = RuntimeError("actor died")

        await manager.stop_cells(["engine-0"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive

    async def test_several_cells_are_stopped_together(self, fake_ray_cluster: FakeRayCluster):
        """A multi-cell suspend is one request, not a caller-side loop."""
        manager = await _launch([_make_spec("engine", num_cells=3)])

        await manager.stop_cells(["engine-0", "engine-2"])

        assert not any(manager.get_cell_infos(pool_ids=["engine"])[c].alive for c in ["engine-0", "engine-2"])

    async def test_an_unknown_cell_id_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        """A typo'd cell id must not silently suspend nothing."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            await manager.stop_cells(["engine-7"])

    async def test_a_cell_that_fails_while_allocating_ports_is_rolled_back_and_can_be_retried(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """A resume dying mid-setup must leave the cell stopped, else every later resume is skipped as a no-op."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])
        fake_ray_cluster.method_errors["_get_node_ip"] = RuntimeError("node gone")

        with pytest.raises(RuntimeError, match="node gone"):
            await manager.start_cells(["engine-0"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert fake_ray_cluster.handles[-1].killed

        del fake_ray_cluster.method_errors["_get_node_ip"]
        await manager.start_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_a_cell_that_fails_its_post_setup_is_rolled_back(self, fake_ray_cluster: FakeRayCluster):
        """Failing after the actors exist must still release them, or the cell stays alive but never serves."""
        command_fails = False

        def launch_command(ctx: LaunchCommandContext) -> str:
            if command_fails:
                raise RuntimeError("cannot render command")
            return "sleep 600"

        manager = await _launch([_make_spec("engine", launch_command=launch_command)])
        await manager.stop_cells(["engine-0"])
        command_fails = True

        with pytest.raises(RuntimeError, match="cannot render command"):
            await manager.start_cells(["engine-0"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert fake_ray_cluster.handles[-1].killed

    async def test_a_failed_start_leaves_the_running_cells_alone(self, fake_ray_cluster: FakeRayCluster):
        """Rollback must be scoped to the cell that failed, not to the siblings that are already serving."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-0"])
        fake_ray_cluster.method_errors["_get_node_ip"] = RuntimeError("node gone")

        with pytest.raises(RuntimeError, match="node gone"):
            await manager.start_cells(["engine-0"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-0"].alive
        assert infos["engine-1"].alive

    async def test_a_restarted_cell_allocates_ports_and_addrs_again(self, fake_ray_cluster: FakeRayCluster):
        """The new process needs its own addresses; a stale addr book would point at the dead one."""
        manager = await _launch([_make_spec("engine")])

        await manager.stop_cells(["engine-0"])
        await manager.start_cells(["engine-0"])

        assert manager.get_worker_addrs("engine-0-0")["primary"] is not None
        assert len(fake_ray_cluster.calls_of("_get_node_ip")) == 2

    async def test_start_runs_every_phase_before_the_next_one(self, fake_ray_cluster: FakeRayCluster):
        """A worker must be addressable before any worker of the cell is handed its command."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        await manager.stop_cells(["engine-0"])
        fake_ray_cluster.calls.clear()

        await manager.start_cells(["engine-0"])

        methods = [call.method for call in fake_ray_cluster.calls]
        assert methods.index("run") > max(i for i, m in enumerate(methods) if m == "_get_node_ip")

    async def test_a_suspended_sibling_does_not_break_the_addr_book(self, fake_ray_cluster: FakeRayCluster):
        """Launch commands are rendered from every spec's addrs, so a suspended cell must drop out of it."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-0"])

        assert [len(addrs) for addrs in manager.get_addrs().values()] == [1]

    async def test_a_cell_restarts_while_a_sibling_stays_suspended(self, fake_ray_cluster: FakeRayCluster):
        """Healing one cell must not depend on every other cell being up."""
        manager = await _launch([_make_spec("engine", num_cells=3)])
        await manager.stop_cells(["engine-0", "engine-1"])

        await manager.start_cells(["engine-1"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-0"].alive
        assert infos["engine-1"].alive and infos["engine-2"].alive

    async def test_worker_addrs_ignore_suspended_cells(self, fake_ray_cluster: FakeRayCluster):
        """Resolving a live worker must not walk into a suspended cell's missing actors."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-0"])

        assert manager.get_worker_addrs("engine-1-0")["primary"] is not None


class TestStartCellsRollback:
    async def test_a_resume_that_fails_midway_leaves_the_cell_suspended(self, fake_ray_cluster: FakeRayCluster):
        """A half-started cell that still looked alive would never be retried by the heal loop."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        async def failing_alloc(self) -> None:
            raise RuntimeError("no ports")

        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(
                    "miles.utils.workers.ray_worker_manager._CommandActorManager.alloc_ports", failing_alloc
                )
                await manager.start_cells(["engine-0"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive

    async def test_a_resume_that_fails_midway_kills_the_actors_it_created(self, fake_ray_cluster: FakeRayCluster):
        """Leaked actors keep holding their GPUs, so the next resume would find no capacity."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        async def failing_post_setup(self) -> None:
            raise RuntimeError("cannot render")

        with pytest.raises(RuntimeError, match="cannot render"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(
                    "miles.utils.workers.ray_worker_manager._CommandActorManager.post_setup", failing_post_setup
                )
                await manager.start_cells(["engine-0"])

        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]

    async def test_the_resume_after_a_failed_one_starts_the_cell_for_real(self, fake_ray_cluster: FakeRayCluster):
        """The whole point of rolling back is that the very next resume attempt is not skipped."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        async def failing_alloc(self) -> None:
            raise RuntimeError("no ports")

        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(
                    "miles.utils.workers.ray_worker_manager._CommandActorManager.alloc_ports", failing_alloc
                )
                await manager.start_cells(["engine-0"])

        await manager.start_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_a_failed_start_rolls_back_the_siblings_of_the_failing_cell(self, fake_ray_cluster: FakeRayCluster):
        """One request is one transaction, so no cell of it may survive half configured."""
        spec = _make_spec("engine", num_cells=2)

        async def failing_post_setup(self) -> None:
            raise RuntimeError("cannot render")

        manager = RayWorkerManager()
        with pytest.raises(RuntimeError, match="cannot render"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(
                    "miles.utils.workers.ray_worker_manager._CommandActorManager.post_setup", failing_post_setup
                )
                await manager.init([spec], {})

        assert not any(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())


class TestSuspendedCellInfos:
    async def test_a_suspended_cell_is_still_described(self, fake_ray_cluster: FakeRayCluster):
        """The api server must still list a suspended cell, or it could never be resumed."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-0"])

        infos = manager.get_cell_infos(pool_ids=["engine"])

        assert sorted(infos) == ["engine-0", "engine-1"]
        assert not infos["engine-0"].alive
        assert infos["engine-1"].alive

    async def test_the_meta_of_a_suspended_cell_is_still_known(self, fake_ray_cluster: FakeRayCluster):
        """Meta comes from the spec, so suspending a cell must not make it unidentifiable."""
        spec = _make_spec("engine").model_copy(update={"meta": lambda ctx: {"model_id": "default"}})
        manager = await _launch([spec])
        await manager.stop_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].meta == {"model_id": "default"}

    async def test_a_suspended_cell_reports_no_workers(self, fake_ray_cluster: FakeRayCluster):
        """Its actors are gone, so anything reading worker names off it must get an empty list."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].worker_names == []


class TestInjectFault:
    async def test_the_fault_reaches_the_selected_worker(self, fake_ray_cluster: FakeRayCluster):
        """A multi-node engine is crashed by crashing one of its node ranks."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        manager.inject_fault("engine-0", mode="sigkill", worker_in_cell_index=1)

        calls = fake_ray_cluster.calls_of("inject_fault")
        assert [call.args for call in calls] == [("sigkill",)]
        assert calls[0].handle is fake_ray_cluster.handles[1]

    async def test_injection_does_not_wait_for_the_worker_to_answer(self, fake_ray_cluster: FakeRayCluster):
        """The worker is about to die, so waiting for its reply would hang the caller."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].failing_methods["inject_fault"] = RuntimeError("actor died")

        manager.inject_fault("engine-0", mode="sigkill", worker_in_cell_index=0)

    async def test_injecting_into_a_suspended_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        """A suspended cell has no worker to crash."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-0"])

        with pytest.raises(RuntimeError, match="not alive"):
            manager.inject_fault("engine-0", mode="sigkill", worker_in_cell_index=0)

    async def test_a_worker_index_beyond_the_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        """Injecting into a neighbouring cell by accident would corrupt the test's premise."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=1)])

        with pytest.raises(IndexError, match="out of range"):
            manager.inject_fault("engine-0", mode="sigkill", worker_in_cell_index=1)

    async def test_a_negative_worker_index_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        """Negative indexing would silently select the last worker instead of failing."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        with pytest.raises(IndexError, match="out of range"):
            manager.inject_fault("engine-0", mode="sigkill", worker_in_cell_index=-1)

    async def test_an_unknown_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        """A typo must not silently inject nothing."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.inject_fault("engine-7", mode="sigkill", worker_in_cell_index=0)
