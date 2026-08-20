from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from tests.fast.utils.workers.fake_ray import EVENT_CREATE, FakeRayCluster

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
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        port_infos=(
            port_infos if port_infos is not None else [PortInfo(name="primary", static_port=8000, allow_dynamic=True)]
        ),
        env_var=lambda: dict(env_var or {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
        ),
        launch_command=launch_command if launch_command is not None else (lambda ctx: "sleep 600"),
    )


async def _launch(
    specs: list[CommandWorkerSpec], pgs: dict[str, PlacementGroupInfo] | None = None
) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(specs, {})
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
            manager.get_worker_addr(f"engine-{cell_index}-{worker_in_cell_index}").port
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

        first = manager.get_worker_addr("engine-0-0").port
        second = manager.get_worker_addr("engine-0-1").port
        assert second >= first + 5
        assert [call.kwargs["count"] for call in fake_ray_cluster.calls_of("_get_free_port_block")] == [5, 5]

    async def test_static_ports_bypass_the_allocator(self, fake_ray_cluster: FakeRayCluster):
        """A port the worker cannot choose is taken from the spec verbatim, without asking the allocator."""
        spec = _make_spec(
            "router",
            port_infos=[
                PortInfo(name="primary", static_port=7777, allow_dynamic=False),
                PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
            ],
        )
        manager = await _launch([spec])

        assert manager.get_worker_addr("router-0-0").port == 7777
        assert len(fake_ray_cluster.calls_of("_get_free_port_block")) == 1
        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [7777]

    async def test_a_static_port_already_taken_on_the_node_fails_the_launch(self, fake_ray_cluster: FakeRayCluster):
        """A stale listener on the pinned port would otherwise be mistaken for our own worker."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)])

        with pytest.raises(AssertionError, match="7777 on 10.0.0.1 is already in use"):
            await _launch([spec])

    async def test_ports_are_tracked_per_node(self, fake_ray_cluster: FakeRayCluster):
        """Workers on different nodes may reuse the same port number."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        first = manager.get_worker_addr("engine-0-0")
        second = manager.get_worker_addr("engine-0-1")
        assert (first.host, second.host) == ("10.0.0.1", "10.0.0.2")
        assert first.port == second.port

    async def test_the_worker_addr_host_is_the_node_the_actor_landed_on(self, fake_ray_cluster: FakeRayCluster):
        """Addresses advertise the actor's own node, not the driver's."""
        fake_ray_cluster.use_node_ips("10.1.2.3")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addr("router-0-0").host == "10.1.2.3"

    async def test_ipv6_hosts_are_bracketed(self, fake_ray_cluster: FakeRayCluster):
        """An ipv6 node address is advertised in url-safe bracketed form."""
        fake_ray_cluster.use_node_ips("2001:db8::7")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addr("router-0-0").host == "[2001:db8::7]"


class TestStaticPortsAreProbedBeforeUse:
    async def test_the_probe_asks_for_the_cell_offset_port_rather_than_the_spec_base(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """Probing the spec's base port would leave every cell but the first checking an address it never binds."""
        spec = _make_spec(
            "session-server",
            num_cells=3,
            port_infos=[PortInfo(name="primary", static_port=7000, allow_dynamic=False, offset_by_cell=True)],
        )
        await _launch([spec])

        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [7000, 7001, 7002]

    async def test_an_occupied_cell_offset_port_refuses_the_launch(self, fake_ray_cluster: FakeRayCluster):
        """A stale listener on a later cell's offset port must be caught too, not only one on the base port."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7002)
        spec = _make_spec(
            "session-server",
            num_cells=3,
            port_infos=[PortInfo(name="primary", static_port=7000, allow_dynamic=False, offset_by_cell=True)],
        )

        with pytest.raises(AssertionError, match="Port 7002 on 10.0.0.1 is already in use"):
            await _launch([spec])

    async def test_the_probe_runs_on_the_node_the_worker_landed_on(self, fake_ray_cluster: FakeRayCluster):
        """A port is only contended on the node that binds it, so the probe must go through the worker's own actor."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        fake_ray_cluster.occupy_ports("10.0.0.2", 7777)
        spec = _make_spec(
            "router",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)],
        )

        with pytest.raises(AssertionError, match="Port 7777 on 10.0.0.2 is already in use"):
            await _launch([spec])

        assert [call.handle.node_ip for call in fake_ray_cluster.calls_of("_is_port_available")] == [
            "10.0.0.1",
            "10.0.0.2",
        ]

    async def test_a_port_occupied_on_an_unrelated_node_does_not_block_the_launch(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """Refusing a port because another machine uses it would make every run hostage to unrelated nodes."""
        fake_ray_cluster.occupy_ports("10.0.0.2", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)])

        manager = await _launch([spec])

        assert manager.get_worker_addrs("router-0-0")["primary"].port == 7777

    async def test_a_master_static_port_is_probed_only_by_the_worker_that_reserves_it(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """Peers that never bind the cell's master port would otherwise refuse a port their own worker 0 is serving."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=3,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=False),
            ],
        )
        await _launch([spec])

        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [9000, 9000]

    async def test_dynamic_ports_are_never_probed_by_the_static_check(self, fake_ray_cluster: FakeRayCluster):
        """The allocator already hands out a free port, and re-probing its own reservation would refuse it."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            ],
        )
        await _launch([spec])

        assert fake_ray_cluster.calls_of("_is_port_available") == []

    async def test_the_refusal_names_the_worker_and_the_endpoint_it_could_not_serve(
        self, fake_ray_cluster: FakeRayCluster
    ):
        """An operator hunting the stale process needs to know which worker wanted which endpoint."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="prometheus", static_port=7777, allow_dynamic=False)])

        with pytest.raises(AssertionError, match="router-0-0 cannot serve its 'prometheus' endpoint"):
            await _launch([spec])

    async def test_a_refused_port_rolls_the_cell_back_before_any_command_runs(self, fake_ray_cluster: FakeRayCluster):
        """A refusal that let the commands start would add our own processes to the stale ones already there."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec(
            "router",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)],
        )

        with pytest.raises(AssertionError):
            await _launch([spec])

        assert fake_ray_cluster.calls_of("run") == []
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]


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
            manager.get_worker_addr(name).port for name in ["router-0-0", "router-0-1", "engine-0-0", "engine-1-0"]
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
                expected = manager.get_worker_addr(f"engine-{cell_index}-{worker_in_cell_index}")
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

        def _env() -> dict[str, str]:
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


class TestGetWorkerAddr:
    async def test_names_are_the_pool_with_the_cell_and_worker_index(self, fake_ray_cluster: FakeRayCluster):
        """Workers are addressable under ``<spec>-<cell>-<worker>``."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])

        for name in ["engine-0-0", "engine-0-1", "engine-1-0", "engine-1-1"]:
            assert manager.get_worker_addr(name).port > 0

    async def test_an_unknown_worker_name_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        """Looking up a worker that does not exist must not silently return an arbitrary address."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        with pytest.raises(AssertionError):
            manager.get_worker_addr("engine-2-0")
