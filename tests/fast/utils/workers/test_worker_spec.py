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
)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8000, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_launch_command_context(**overrides) -> LaunchCommandContext:
    kwargs = dict(
        cell_index=0,
        worker_in_cell_index=0,
        gpu_ids=[],
        self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
        pool_addrs={},
    )
    kwargs.update(overrides)
    return LaunchCommandContext(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="demo-worker",
        port_infos=[_make_port_info()],
        env_var=lambda: {"DEMO": "1"},
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

        def env_var() -> dict[str, str]:
            calls.append(1)
            return {"A": "b"}

        spec = BaseWorkerSpec(**_make_base_kwargs(env_var=env_var))
        assert calls == []
        assert spec.env_var() == {"A": "b"}

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            BaseWorkerSpec(**_make_base_kwargs(unknown_field=1))

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        with pytest.raises(ValidationError):
            spec.name = "other"


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
            ctor_kwargs=lambda: {},
        )
        assert spec.worker_class == "miles.ray.rollout.inference_controller.InferenceController"
        assert isinstance(spec, BaseWorkerSpec)

    def test_ctor_kwargs_is_stored_uncalled(self):
        """The ctor_kwargs callable is stored as-is and only evaluated on demand."""
        calls = []

        def ctor_kwargs() -> dict:
            calls.append(1)
            return {"x": 1}

        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=ctor_kwargs,
        )
        assert calls == []
        assert spec.ctor_kwargs() == {"x": 1}


class TestServeWorkerSpecRpcPortInjection:
    def _make_spec(self, **overrides) -> ServeWorkerSpec:
        return ServeWorkerSpec(
            **_make_base_kwargs(**overrides),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda: {},
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
