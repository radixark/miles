from typing import Any, ClassVar, Self

import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec

from miles.backends.sglang_utils.sglang_config import SglangScalingConfig
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.args.schema import BaseConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import _assert_worker_ports_fit
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import (
    DEFAULT_RPC_PORT,
    RPC_PORT_NAME,
    BaseCommandSpec,
    BaseServeSpec,
    BaseSpec,
    HostAndPort,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    StaticMeta,
    WorkerCtorContext,
    WorkerLaunchContext,
)

_SCHEDULING = SchedulingSpec(num_cells=2, num_workers_per_cell=4, num_gpus_per_worker=0.4)


class _DemoLeafConfig(BaseLeafConfig):
    demo_flag: int


class _DemoRunConfig(BaseConfig):
    demo_flag: int
    unrelated_flag: str


class _PlainCommandSpec(BaseCommandSpec):
    @classmethod
    def create(cls, config: Any) -> Self:
        return cls(args=config, name="demo-command", port_infos=[])

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return _SCHEDULING

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        return "sleep 1"


class _PlainServeSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "demo"
    config_class: ClassVar[type[BaseLeafConfig]] = _DemoLeafConfig
    worker_class: str = "miles.demo.Worker"

    @classmethod
    def create(cls, config: _DemoLeafConfig) -> Self:
        return cls(args=config, name="demo-serve")

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return _SCHEDULING

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return {}


def _make_scaling() -> ScalingConfig:
    return ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))


def _make_launch_context(**overrides) -> WorkerLaunchContext:
    kwargs = dict(args=None, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[])
    kwargs.update(overrides)
    return WorkerLaunchContext(**kwargs)


def _make_launch_command_context(**overrides) -> LaunchCommandContext:
    kwargs = dict(
        args=None,
        cell_index=0,
        worker_in_cell_index=0,
        num_workers_per_cell=1,
        gpu_ids=[],
        local_gpu_ids=[],
        self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
        pool_addrs={},
    )
    kwargs.update(overrides)
    return LaunchCommandContext(**kwargs)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8080, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    kwargs = dict(name="demo-worker", port_infos=[_make_port_info()], fixed_scheduling=_SCHEDULING)
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


class TestPortInfoEffectiveStaticPort:
    def test_offsets_a_per_worker_port_by_the_whole_block_of_the_workers_before_it(self):
        """Offsetting by the bare index hands a worker an address inside the previous worker's block."""
        port_info = _make_port_info(static_port=8080, mode="per_worker", num_consecutive=4)

        assert port_info.effective_static_port(worker_in_pod_index=2) == 8088

    def test_leaves_a_master_port_where_every_worker_of_the_pod_expects_it(self):
        """A master port names one endpoint the whole pod talks to, so shifting it per worker would split them."""
        port_info = _make_port_info(static_port=8080, mode="master", num_consecutive=4)

        assert port_info.effective_static_port(worker_in_pod_index=2) == 8080


class TestPortInfoCellOffset:
    def test_a_dynamically_allocated_port_cannot_be_offset_by_cell(self):
        """A cell offset applied to a port whose number is chosen at runtime would point at an unrelated socket."""
        with pytest.raises(ValidationError, match="cannot be offset by cell index"):
            _make_port_info(offset_by_cell=True, allow_dynamic=True)

    def test_a_pinned_port_may_be_offset_by_cell(self):
        """Pinned ports are the only ones a cell offset can meaningfully shift, so that combination stays legal."""
        port_info = _make_port_info(static_port=5100, offset_by_cell=True, allow_dynamic=False)

        assert (port_info.static_port, port_info.offset_by_cell) == (5100, True)


class TestBaseSpec:
    def test_the_all_selector_cannot_be_stored_as_a_pool_component(self) -> None:
        """A worker pool must name one concrete deployment component rather than the all-components selector."""
        concrete = FakeCommandSpec(**_make_base_kwargs(deploy_component=DeployComponent.TRAINER), command=str)

        assert concrete.deploy_component is DeployComponent.TRAINER
        with pytest.raises(ValidationError, match="must name the one component.*not the selector"):
            FakeCommandSpec(**_make_base_kwargs(deploy_component=DeployComponent.ALL), command=str)

    def test_constructs_and_exposes_fields(self):
        """A spec keeps its name and ports as provided."""
        spec = _PlainCommandSpec(args=None, name="demo-worker", port_infos=[_make_port_info()])
        assert spec.name == "demo-worker"
        assert spec.port_infos[0].static_port == 8080

    def test_a_spec_that_declares_no_env_sets_none(self):
        """Env vars are opt-in per spec, so the base hook must add nothing to the worker's environment."""
        spec = _PlainCommandSpec.create(None)

        assert spec.env_var(_make_launch_context()) == {}

    def test_a_spec_that_declares_no_meta_publishes_none_for_any_cell(self):
        """Cell meta is opt-in, so an undeclared one must resolve to nothing rather than leak another cell's."""
        spec = _PlainCommandSpec.create(None)

        assert spec.static_meta == StaticMeta()
        assert spec.static_meta.resolve(cell_index=3) == {}

    def test_an_incomplete_spec_cannot_be_instantiated(self):
        """A spec missing its scheduling or creation hooks would only fail once a worker is launched."""
        with pytest.raises(TypeError, match="abstract"):
            BaseSpec(args=None, name="demo-worker", port_infos=[])

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            _PlainCommandSpec(args=None, name="demo-worker", port_infos=[], unknown_field=1)

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        spec = _PlainCommandSpec.create(None)
        with pytest.raises(ValidationError):
            spec.name = "other"


class TestLaunchCommandContext:
    def test_the_context_refuses_to_be_built_without_local_gpu_ids(self):
        """A default here would let a manager that never probed the worker launch it against the wrong devices."""
        kwargs = dict(
            args=None,
            cell_index=0,
            worker_in_cell_index=0,
            num_workers_per_cell=1,
            gpu_ids=[],
            self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            pool_addrs={},
        )

        with pytest.raises(ValidationError):
            LaunchCommandContext(**kwargs)


class TestBaseCommandSpec:
    def test_a_command_spec_is_handed_the_whole_run_config(self):
        """A command pool renders its argv from the run's own flags, so slicing must not drop any of them."""
        run_config = _DemoRunConfig(demo_flag=3, unrelated_flag="kept")

        assert _PlainCommandSpec.slice_configs(run_config) == [run_config]

    def test_a_command_spec_must_say_how_to_launch_its_workers(self):
        """Without a launch command the manager would have nothing to run in the worker's actor."""

        class _NoLaunchCommand(BaseCommandSpec):
            @classmethod
            def create(cls, config: Any) -> Self:
                return cls(args=config, name="demo-command", port_infos=[])

            def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
                return _SCHEDULING

        with pytest.raises(TypeError, match="launch_command"):
            _NoLaunchCommand.create(None)


class TestBaseServeSpec:
    def test_constructs_with_worker_class(self):
        """A serve spec carries the worker class path besides base fields."""
        spec = _PlainServeSpec.create(_DemoLeafConfig(demo_flag=1))

        assert spec.worker_class == "miles.demo.Worker"
        assert isinstance(spec, BaseSpec)

    def test_a_serve_spec_is_handed_only_its_own_leaf_config(self):
        """A served pool is rebuilt from the config it was sliced, so flags outside its leaf must not reach it."""
        (config,) = _PlainServeSpec.slice_configs(_DemoRunConfig(demo_flag=3, unrelated_flag="dropped"))

        assert config == _DemoLeafConfig(demo_flag=3)

    def test_a_serve_spec_must_say_how_to_build_its_worker(self):
        """Without constructor kwargs a pod could not build the worker class it names."""

        class _NoCtorKwargs(BaseServeSpec):
            worker_type: ClassVar[str] = "demo"
            config_class: ClassVar[type[BaseLeafConfig]] = _DemoLeafConfig

            @classmethod
            def create(cls, config: Any) -> Self:
                return cls(args=config, name="demo-serve", worker_class="miles.demo.Worker")

            def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
                return _SCHEDULING

        with pytest.raises(TypeError, match="ctor_kwargs"):
            _NoCtorKwargs.create(None)


class TestBaseServeSpecRpcPort:
    def _make_spec(self, **overrides) -> BaseServeSpec:
        return FakeServeSpec(**_make_base_kwargs(**overrides), worker_class="miles.demo.Worker")

    def test_a_serve_spec_that_declares_no_ports_exposes_the_default_rpc_port(self):
        """Every serve worker runs the rpc server, so a spec with no port declaration must still get one."""
        spec = _PlainServeSpec.create(_DemoLeafConfig(demo_flag=1))

        (rpc,) = [port_info for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
        assert rpc.static_port == DEFAULT_RPC_PORT
        assert rpc.mode == "per_worker"
        assert rpc.allow_dynamic is True

    def test_declared_ports_are_kept_exactly_as_declared(self):
        """A spec that lists its ports owns the list, including where its rpc port sits in it."""
        http = _make_port_info()
        explicit = PortInfo(name=RPC_PORT_NAME, static_port=9999, mode="per_worker", allow_dynamic=False)

        spec = self._make_spec(port_infos=[http, explicit])

        assert spec.port_infos == [http, explicit]

    def test_an_explicit_rpc_port_given_as_a_dict_is_kept(self):
        """Callers may declare ports as raw dicts, and such a declaration must still yield the declared rpc port."""
        spec = self._make_spec(port_infos=[dict(name=RPC_PORT_NAME, static_port=9999)])

        assert spec.port_infos == [PortInfo(name=RPC_PORT_NAME, static_port=9999)]

    def test_command_specs_get_no_rpc_port(self):
        """Only serve workers run the rpc server, so only they get the port."""
        command = FakeCommandSpec(**_make_base_kwargs(), command=str)
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

        _assert_worker_ports_fit(spec, scaling=_make_scaling())

    def test_rejects_rank_ports_reaching_into_another_port(self):
        """Rank 2 would bind the master port and every collective would rendezvous on nothing."""
        spec = _serve_spec(
            num_gpus_per_node=4,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8002)],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            _assert_worker_ports_fit(spec, scaling=_make_scaling())

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
            _assert_worker_ports_fit(spec, scaling=_make_scaling())

    def test_a_port_below_the_rpc_port_is_untouched(self):
        """Ranks climb upwards only, so a lower port can never be reached."""
        spec = _serve_spec(
            num_gpus_per_node=8,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=7000)],
        )

        _assert_worker_ports_fit(spec, scaling=_make_scaling())

    def test_a_pod_of_one_rank_needs_only_its_own_rpc_port(self):
        """Nodes as wide as a cell put one rank in each pod, which must not be constrained by neighbours."""
        spec = _serve_spec(
            num_gpus_per_node=1,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8001)],
        )

        _assert_worker_ports_fit(spec, scaling=_make_scaling())


def _serve_spec(*, num_gpus_per_node: int, **overrides) -> BaseServeSpec:
    scheduling = SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=8,
        num_gpus_per_worker=1,
        num_gpu_slots_per_worker=1,
        num_gpus_per_node=num_gpus_per_node,
    )
    return FakeServeSpec(
        **_make_base_kwargs(fixed_scheduling=scheduling, **overrides), worker_class="miles.demo.Worker"
    )


class TestBaseServeSpecExtraScheduling:
    def test_concurrency_groups_default_to_absent(self):
        """Most workers need no concurrency groups, so the field stays optional."""
        spec = _PlainServeSpec.create(_DemoLeafConfig(demo_flag=1))

        assert spec.concurrency_groups is None

    def test_concurrency_groups_are_carried_on_the_spec(self):
        """The trainer needs its heartbeat rpc served outside the default group."""
        spec = FakeServeSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            concurrency_groups={"heartbeat_status": 1, "default": 1},
        )

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1}


class TestLaunchCommandContextPoolAddrs:
    def test_a_launch_command_reads_a_peer_address_out_of_the_pool_keyed_map(self):
        """A command renders a peer's address by looking that peer's pool id up in pool_addrs."""
        spec = FakeCommandSpec(
            **_make_base_kwargs(),
            command=lambda ctx: f"serve --backend {ctx.pool_addrs['inference-router-0'][0]['primary'].addr}",
        )
        ctx = _make_launch_command_context(
            pool_addrs={"inference-router-0": [{"primary": HostAndPort(host="10.0.0.1", port=3000)}]}
        )

        assert spec.launch_command(ctx) == "serve --backend http://10.0.0.1:3000"

    def test_the_map_of_a_pool_with_several_workers_keeps_every_worker_under_that_one_key(self):
        """One key per pool, listing all of its workers, is what lets a command address a whole pool."""
        workers = [
            {"primary": HostAndPort(host="10.0.0.1", port=3000)},
            {"primary": HostAndPort(host="10.0.0.2", port=3000)},
        ]
        ctx = _make_launch_command_context(pool_addrs={"session-server": workers})

        assert list(ctx.pool_addrs) == ["session-server"]
        assert ctx.pool_addrs["session-server"] == workers

    def test_the_legacy_spec_keyed_name_is_not_accepted_beside_the_pool_keyed_one(self):
        """Two accepted names for one map would let the retired spec_addrs vocabulary creep back in unnoticed."""
        with pytest.raises(ValidationError):
            _make_launch_command_context(spec_addrs={})

    def test_a_context_without_the_pool_map_is_rejected(self):
        """Making the map optional would let a caller that forgot to wire it render commands against nothing."""
        with pytest.raises(ValidationError):
            LaunchCommandContext(
                args=None,
                cell_index=0,
                worker_in_cell_index=0,
                num_workers_per_cell=1,
                gpu_ids=[],
                self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            )
