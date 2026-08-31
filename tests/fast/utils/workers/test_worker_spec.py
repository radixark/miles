import pytest
from pydantic import ValidationError

from miles.utils.workers.serving import serve_inner
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
    WorkerLaunchContext,
)


def _make_launch_context(**overrides) -> WorkerLaunchContext:
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[])
    kwargs.update(overrides)
    return WorkerLaunchContext(**kwargs)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8000, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_launch_command_context(**overrides) -> LaunchCommandContext:
    kwargs = dict(
        cell_index=0,
        worker_in_cell_index=0,
        gpu_ids=[],
        local_gpu_ids=[],
        self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
        pool_addrs={},
    )
    kwargs.update(overrides)
    return LaunchCommandContext(**kwargs)


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
        assert spec.port_infos[0].static_port == 8000
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


class TestLaunchCommandContext:
    def test_the_context_refuses_to_be_built_without_local_gpu_ids(self):
        """A default here would let a manager that never probed the worker launch it against the wrong devices."""
        kwargs = dict(
            cell_index=0,
            worker_in_cell_index=0,
            gpu_ids=[],
            self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            pool_addrs={},
        )

        with pytest.raises(ValidationError):
            LaunchCommandContext(**kwargs)


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
        assert spec.ctor_kwargs(_make_launch_context()) == {"x": 1}


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

    def test_the_injected_port_is_the_one_the_serve_entrypoint_binds_by_default(self):
        """A spec advertising a port its own process does not bind leaves every caller talking to nothing."""
        assert DEFAULT_RPC_PORT == serve_inner.DEFAULT_PORT


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
            method_concurrency_groups={"get_heartbeat_status": "heartbeat_status"},
        )

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1}
        assert spec.method_concurrency_groups == {"get_heartbeat_status": "heartbeat_status"}

    def test_groups_without_routed_methods_are_rejected(self):
        """Threading the actor while every method stays in the default group buys nothing."""
        with pytest.raises(ValidationError, match="together"):
            ServeWorkerSpec(
                **_make_base_kwargs(),
                worker_class="miles.demo.Worker",
                ctor_kwargs=lambda _ctx: {},
                concurrency_groups={"heartbeat_status": 1, "default": 1},
            )

    def test_routed_methods_without_groups_are_rejected(self):
        """Ray rejects an actor whose method names a concurrency group the class never declares."""
        with pytest.raises(ValidationError, match="together"):
            ServeWorkerSpec(
                **_make_base_kwargs(),
                worker_class="miles.demo.Worker",
                ctor_kwargs=lambda _ctx: {},
                method_concurrency_groups={"get_heartbeat_status": "heartbeat_status"},
            )

    def test_a_method_routed_to_an_undeclared_group_is_rejected(self):
        """Ray rejects the actor at creation time, long after the spec could have said why."""
        with pytest.raises(ValidationError, match="undeclared concurrency groups"):
            ServeWorkerSpec(
                **_make_base_kwargs(),
                worker_class="miles.demo.Worker",
                ctor_kwargs=lambda _ctx: {},
                concurrency_groups={"default": 1},
                method_concurrency_groups={"get_heartbeat_status": "heartbeat_status"},
            )

    def test_a_declared_group_nobody_routes_to_is_allowed(self):
        """The trainer declares a default group precisely because no method is routed to it."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
            concurrency_groups={"heartbeat_status": 1, "default": 1, "kill_self": 1},
            method_concurrency_groups={"get_heartbeat_status": "heartbeat_status"},
        )

        assert set(spec.concurrency_groups) - set(spec.method_concurrency_groups.values()) == {"default", "kill_self"}

    def test_the_rejection_names_the_worker_and_every_undeclared_group(self):
        """A message listing the declared groups instead of the missing ones sends the reader the wrong way."""
        with pytest.raises(ValidationError, match=r"'demo-worker'.*\['fault_injector', 'kill_self'\]"):
            ServeWorkerSpec(
                **_make_base_kwargs(),
                worker_class="miles.demo.Worker",
                ctor_kwargs=lambda _ctx: {},
                concurrency_groups={"heartbeat_status": 1, "default": 1},
                method_concurrency_groups={
                    "get_heartbeat_status": "heartbeat_status",
                    "kill_self": "kill_self",
                    "inject_fault": "fault_injector",
                },
            )

    def test_ctor_kwargs_receive_the_worker_position(self):
        """Each worker needs its own rank, so the callable is per worker."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda ctx: {"rank": ctx.worker_in_cell_index, "gpu_ids": ctx.gpu_ids},
        )

        kwargs = spec.ctor_kwargs(_make_launch_context(worker_in_cell_index=3, gpu_ids=[2]))

        assert kwargs == {"rank": 3, "gpu_ids": [2]}


class TestLaunchCommandContextPoolAddrs:
    def test_a_launch_command_reads_a_peer_address_out_of_the_pool_keyed_map(self):
        """A command renders a peer's address by looking that peer's pool id up in pool_addrs."""
        spec = CommandWorkerSpec(
            **_make_base_kwargs(),
            launch_command=lambda ctx: f"serve --backend {ctx.pool_addrs['inference-router-0'][0]['primary'].addr}",
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
                cell_index=0,
                worker_in_cell_index=0,
                gpu_ids=[],
                self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            )
