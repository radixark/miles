import pytest
from pydantic import ValidationError
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import _assert_worker_ports_fit
from miles.utils.workers.worker_spec import (
    DEFAULT_RPC_PORT,
    RPC_PORT_NAME,
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)


def _make_launch_context(**overrides) -> WorkerLaunchContext:
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[])
    kwargs.update(overrides)
    return WorkerLaunchContext(**kwargs)


def _make_launch_command_context(**overrides) -> LaunchCommandContext:
    kwargs = dict(
        cell_index=0,
        worker_in_cell_index=0,
        gpu_ids=[],
        self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
        spec_addrs={},
    )
    kwargs.update(overrides)
    return LaunchCommandContext(**kwargs)


def _make_ctor_context(**overrides) -> WorkerCtorContext:
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=FakeBackendCapability())
    kwargs.update(overrides)
    return WorkerCtorContext(**kwargs)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8080, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="demo-worker",
        port_infos=[_make_port_info()],
        env_var=lambda _ctx: {"DEMO": "1"},
        scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=4, num_gpus_per_worker=0.4),
    )
    kwargs.update(overrides)
    return kwargs


class TestPortInfo:
    def test_accepts_both_modes(self):
        """Both per_worker and master are valid modes."""
        assert _make_port_info(mode="per_worker").mode == "per_worker"
        assert _make_port_info(mode="master").mode == "master"

    def test_rejects_unknown_mode(self):
        """An unknown mode literal is rejected."""
        with pytest.raises(ValidationError):
            _make_port_info(mode="broadcast")

    def test_num_consecutive_defaults_to_one(self):
        """A port reserves a single slot unless a block is requested."""
        assert _make_port_info().num_consecutive == 1
        assert _make_port_info(num_consecutive=32).num_consecutive == 32

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            _make_port_info(unknown_field=1)

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        port_info = _make_port_info()
        with pytest.raises(ValidationError):
            port_info.static_port = 9000


class TestPortInfoCellOffset:
    def test_a_dynamically_allocated_port_cannot_be_offset_by_cell(self):
        """A cell offset applied to a port whose number is chosen at runtime would point at an unrelated socket."""
        with pytest.raises(ValidationError, match="cannot be offset by cell index"):
            _make_port_info(offset_by_cell=True, allow_dynamic=True)

    def test_a_pinned_port_may_be_offset_by_cell(self):
        """Pinned ports are the only ones a cell offset can meaningfully shift, so that combination stays legal."""
        port_info = _make_port_info(static_port=5100, offset_by_cell=True, allow_dynamic=False)

        assert (port_info.static_port, port_info.offset_by_cell) == (5100, True)


class TestBaseWorkerSpec:
    def test_constructs_and_exposes_fields(self):
        """A spec keeps its name, ports, and scheduling as provided."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        assert spec.name == "demo-worker"
        assert spec.port_infos[0].static_port == 8080
        assert spec.scheduling.num_cells == 2

    def test_env_var_is_stored_uncalled(self):
        """The env_var callable is stored as-is and only evaluated on demand."""
        calls = []

        def env_var(_ctx) -> dict[str, str]:
            calls.append(1)
            return {"A": "b"}

        spec = BaseWorkerSpec(**_make_base_kwargs(env_var=env_var))
        assert calls == []
        assert spec.env_var(_make_launch_context()) == {"A": "b"}

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            BaseWorkerSpec(**_make_base_kwargs(unknown_field=1))

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        with pytest.raises(ValidationError):
            spec.name = "other"

    def test_delete_permission_is_not_combined_with_the_read_only_capability(self):
        """Pod deletion already includes reads, so combining both capabilities would choose an ambiguous account."""
        with pytest.raises(ValidationError, match="both platform read and delete"):
            BaseWorkerSpec(
                **_make_base_kwargs(),
                needs_platform_read_permission=True,
                needs_platform_delete_permission=True,
            )


class TestCommandWorkerSpec:
    def test_constructs_with_launch_command(self):
        """A command spec carries the launch command callable besides base fields."""
        spec = CommandWorkerSpec(**_make_base_kwargs(), launch_command=lambda ctx: "python -m sglang.launch_server")
        ctx = _make_launch_command_context()
        assert spec.launch_command(ctx) == "python -m sglang.launch_server"
        assert isinstance(spec, BaseWorkerSpec)

    def test_launch_command_is_stored_uncalled(self):
        """The launch_command callable is only evaluated once a context is available."""
        calls: list[LaunchCommandContext] = []

        def launch_command(ctx: LaunchCommandContext) -> str:
            calls.append(ctx)
            http = ctx.self_addrs["http"]
            return f"serve --host {http.host} --port {http.port}"

        spec = CommandWorkerSpec(**_make_base_kwargs(), launch_command=launch_command)
        assert calls == []

        ctx = _make_launch_command_context(self_addrs={"http": HostAndPort(host="10.0.0.1", port=9001)})
        assert spec.launch_command(ctx) == "serve --host 10.0.0.1 --port 9001"
        assert calls == [ctx]


class TestServeWorkerSpec:
    def test_constructs_with_worker_class(self):
        """A serve spec carries the worker class path besides base fields."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.ray.rollout.inference_controller.InferenceController",
            ctor_kwargs=lambda _ctx: {},
        )
        assert spec.worker_class == "miles.ray.rollout.inference_controller.InferenceController"
        assert isinstance(spec, BaseWorkerSpec)

    def test_ctor_kwargs_is_stored_uncalled(self):
        """The ctor_kwargs callable is stored as-is and only evaluated on demand."""
        calls = []

        def ctor_kwargs(_ctx) -> dict:
            calls.append(1)
            return {"x": 1}

        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=ctor_kwargs,
        )
        assert calls == []
        assert spec.ctor_kwargs(_make_ctor_context()) == {"x": 1}


class TestServeWorkerSpecRpcPortInjection:
    def _make_spec(self, **overrides) -> ServeWorkerSpec:
        return ServeWorkerSpec(
            **_make_base_kwargs(**overrides),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
        )

    def test_rpc_port_is_injected_by_default(self):
        """Every serve worker automatically exposes an rpc port."""
        spec = self._make_spec()
        (rpc,) = [port_info for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
        assert rpc.static_port == DEFAULT_RPC_PORT
        assert rpc.mode == "per_worker"
        assert rpc.allow_dynamic is True

    def test_injection_keeps_declared_ports(self):
        """The injected rpc port is appended after the declared ports."""
        spec = self._make_spec()
        assert [port_info.name for port_info in spec.port_infos] == ["http", RPC_PORT_NAME]

    def test_explicit_rpc_port_is_not_duplicated(self):
        """An explicitly declared rpc port wins over the injected default."""
        explicit = PortInfo(name=RPC_PORT_NAME, static_port=9999, mode="per_worker", allow_dynamic=False)
        spec = self._make_spec(port_infos=[explicit])
        assert spec.port_infos == [explicit]

    def test_an_explicit_rpc_port_given_as_a_dict_is_not_duplicated(self):
        """Callers may declare ports as raw dicts, and such a declaration must still suppress the injected default."""
        spec = self._make_spec(port_infos=[dict(name=RPC_PORT_NAME, static_port=9999)])

        assert spec.port_infos == [PortInfo(name=RPC_PORT_NAME, static_port=9999)]

    def test_ports_given_as_dicts_still_receive_the_injected_rpc_port(self):
        """Reading a dict port's name must not be confused with reading the rpc name, or the rpc port goes missing."""
        spec = self._make_spec(port_infos=[dict(name="http", static_port=8080)])

        assert [port_info.name for port_info in spec.port_infos] == ["http", RPC_PORT_NAME]

    def test_base_and_command_specs_get_no_rpc_port(self):
        """Only serve workers run the rpc server, so only they get the port."""
        base = BaseWorkerSpec(**_make_base_kwargs())
        command = CommandWorkerSpec(**_make_base_kwargs(), launch_command=lambda ctx: "sleep 1")
        assert RPC_PORT_NAME not in [port_info.name for port_info in base.port_infos]
        assert RPC_PORT_NAME not in [port_info.name for port_info in command.port_infos]


class TestSchedulingSpecPinToHead:
    def test_workers_are_not_pinned_to_the_head_node_by_default(self):
        """Pinning is opt-in, otherwise every worker of every spec would crowd onto the head node."""
        assert SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0).pin_to_head is False
        assert SchedulingSpec.single(num_gpus_per_worker=0).pin_to_head is False

    def test_the_single_worker_shortcut_forwards_the_pin_flag(self):
        """The convenience constructor must not silently drop the pin request."""
        scheduling = SchedulingSpec.single(num_gpus_per_worker=0.5, pin_to_head=True)

        assert (scheduling.num_cells, scheduling.num_workers_per_cell) == (1, 1)
        assert scheduling.num_gpus_per_worker == 0.5
        assert scheduling.pin_to_head is True


class TestSchedulingSpecPodPacking:
    def test_a_cell_a_node_can_hold_rides_in_one_pod(self):
        """A cell no bigger than a node must not be spread, however many workers it holds."""
        scheduling = _gpu_scheduling(num_workers_per_cell=8, num_gpus_per_node=8)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (1, 8)

    def test_a_cell_spanning_several_nodes_is_tiled_by_them(self):
        """This is the whole point of the derivation: 16 gpus on 8-gpu nodes are two equal pods."""
        scheduling = _gpu_scheduling(num_workers_per_cell=16, num_gpus_per_node=8)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (2, 8)

    def test_a_cell_that_claims_no_gpu_rides_in_one_pod(self):
        """A cpu spec has no node shape to tile, so its whole cell travels together."""
        scheduling = SchedulingSpec(num_cells=1, num_workers_per_cell=4, num_gpus_per_worker=0)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (1, 4)

    def test_rejects_a_gpu_cell_that_never_says_how_big_a_node_is(self):
        """Forgetting the node shape used to pack one rank per pod in silence."""
        scheduling = _gpu_scheduling(num_workers_per_cell=8, num_gpus_per_node=0)

        with pytest.raises(AssertionError, match="divide 8 by zero"):
            scheduling.pods_per_cell()

    def test_rejects_a_cell_that_is_not_a_whole_number_of_nodes(self):
        """A trailing partial node would leave the last pod fewer gpus than its ranks need."""
        scheduling = _gpu_scheduling(num_workers_per_cell=12, num_gpus_per_node=8)

        with pytest.raises(AssertionError, match="12 is not a whole number of 8"):
            scheduling.pods_per_cell()

    def test_rejects_a_cell_whose_workers_cannot_tile_its_pods(self):
        """A trailing partial pod would shift every later worker's name and rpc port."""
        scheduling = SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=2,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=12,
            num_gpus_per_node=8,
        )

        with pytest.raises(AssertionError, match="2 is not a whole number of 3"):
            scheduling.workers_per_pod()


def _gpu_scheduling(*, num_workers_per_cell: int, num_gpus_per_node: int) -> SchedulingSpec:
    return SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=num_workers_per_cell,
        num_gpus_per_worker=1,
        num_gpu_slots_per_worker=1,
        num_gpus_per_node=num_gpus_per_node,
    )


class TestAssertRankPortsFit:
    def test_ranks_sharing_a_pod_may_climb_up_to_the_next_port_block(self):
        """The ports a pod hands its ranks are free, so the spec must be accepted."""
        spec = _serve_spec(
            num_gpus_per_node=4,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8004)],
        )

        _assert_worker_ports_fit(spec)

    def test_rejects_rank_ports_reaching_into_another_port(self):
        """Rank 2 would bind the master port and every collective would rendezvous on nothing."""
        spec = _serve_spec(
            num_gpus_per_node=4,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8002)],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            _assert_worker_ports_fit(spec)

    def test_rejects_rank_ports_reaching_into_a_consecutive_port_block(self):
        """A block claims num_consecutive ports, so the collision test must span all of them."""
        spec = _serve_spec(
            num_gpus_per_node=8,
            port_infos=[
                PortInfo(name=RPC_PORT_NAME, static_port=8000),
                PortInfo(name="dist_init", static_port=8003, num_consecutive=30),
            ],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            _assert_worker_ports_fit(spec)

    def test_a_port_below_the_rpc_port_is_untouched(self):
        """Ranks climb upwards only, so a lower port can never be reached."""
        spec = _serve_spec(
            num_gpus_per_node=8,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=7000)],
        )

        _assert_worker_ports_fit(spec)

    def test_a_pod_of_one_rank_needs_only_its_own_rpc_port(self):
        """Nodes as wide as a cell put one rank in each pod, which must not be constrained by neighbours."""
        spec = _serve_spec(
            num_gpus_per_node=1,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8001)],
        )

        _assert_worker_ports_fit(spec)


def _serve_spec(*, num_gpus_per_node: int, **overrides) -> ServeWorkerSpec:
    scheduling = SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=8,
        num_gpus_per_worker=1,
        num_gpu_slots_per_worker=1,
        num_gpus_per_node=num_gpus_per_node,
    )
    return ServeWorkerSpec(
        **_make_base_kwargs(scheduling=scheduling, **overrides),
        worker_class="miles.demo.Worker",
        ctor_kwargs=lambda _ctx: {},
    )


class TestServeWorkerSpecExtraScheduling:
    def test_concurrency_groups_default_to_absent(self):
        """Most workers need no concurrency groups, so the field stays optional."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
        )

        assert spec.concurrency_groups is None

    def test_concurrency_groups_are_carried_on_the_spec(self):
        """The trainer needs its heartbeat rpc served outside the default group."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
            concurrency_groups={"heartbeat_status": 1, "default": 1},
        )

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1}

    def test_ctor_kwargs_receive_the_worker_position(self):
        """Each worker needs its own rank, so the callable is per worker."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda ctx: {"rank": ctx.worker_in_cell_index, "gpu_ids": ctx.gpu_ids},
        )

        kwargs = spec.ctor_kwargs(_make_ctor_context(worker_in_cell_index=3, gpu_ids=[2]))

        assert kwargs == {"rank": 3, "gpu_ids": [2]}
