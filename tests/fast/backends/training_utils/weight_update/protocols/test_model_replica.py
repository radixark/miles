import dataclasses
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import msgspec
import pytest
import torch


@dataclasses.dataclass
class _Parallelism:
    tp_rank: int
    global_rank: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _ServerArgs:
    quantization: str | None = None
    moe_runner_backend: str = "auto"
    enable_dp_lm_head: bool = False
    port: int = 30000
    mem_fraction_static: float = 0.8

    def __getattr__(self, name: str) -> None:
        # the other server args a replica key reads, all unset
        if name.startswith("_"):
            raise AttributeError(name)
        return None


def _config(model_replica_module: ModuleType, *, tp_rank: int, global_rank: int, **server_args):
    return model_replica_module.RolloutEngineRankConfig(
        parallelism=_Parallelism(tp_rank=tp_rank, global_rank=global_rank), server_args=_ServerArgs(**server_args)
    )


class TestShardLayoutKey:
    def test_ranks_that_differ_only_in_their_launch_or_serving_share_a_layout(
        self, model_replica_module: ModuleType
    ) -> None:
        """Engines on other GPUs, ports or memory settings, such as the two PD roles, hold the same rank the same way,
        so one replica serves them all."""
        first_engine = _config(model_replica_module, tp_rank=0, global_rank=0, port=30000, mem_fraction_static=0.8)
        second_engine = _config(model_replica_module, tp_rank=0, global_rank=8, port=30001, mem_fraction_static=0.6)

        assert first_engine.shard_layout_key == second_engine.shard_layout_key

    def test_ranks_with_another_shard_or_sglang_arguments_do_not(self, model_replica_module: ModuleType) -> None:
        """A replica built for one shard, quantization, MoE backend or lm_head sharding would write wrong bytes into
        another."""
        config = _config(model_replica_module, tp_rank=0, global_rank=0)

        assert config.shard_layout_key != _config(model_replica_module, tp_rank=1, global_rank=1).shard_layout_key
        for server_args in ({"quantization": "fp8"}, {"moe_runner_backend": "triton"}, {"enable_dp_lm_head": True}):
            assert (
                config.shard_layout_key
                != _config(model_replica_module, tp_rank=0, global_rank=0, **server_args).shard_layout_key
            )


class TestModelReplicas:
    @pytest.fixture
    def model_replicas_of_width(self, model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch):
        """`ModelReplicas` whose replica for tp rank r is a linear layer of `widths[r]` inputs."""

        def make(widths: dict[int, int]):
            monkeypatch.setattr(
                model_replica_module,
                "_build_cpu_replica",
                lambda config, model_path: torch.nn.Linear(widths[config.parallelism.tp_rank], 1, bias=False),
            )
            monkeypatch.setattr(
                model_replica_module, "ParameterMapper", SimpleNamespace(from_model=lambda model: None)
            )
            # CPU CI has no pinned memory
            monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
            return model_replica_module.ModelReplicas(model_path="/model")

        return make

    def test_a_replica_for_another_layout_loads_into_the_shared_buffer(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """Writes read the shared buffer, so a later replica's loads must land in it."""
        model_replicas = model_replicas_of_width({0: 4, 1: 4})

        first = model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))
        second = model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))

        assert second is not first
        assert (
            second.weight.data_ptr()
            == first.weight.data_ptr()
            == model_replicas.shared_params_dict["weight"].data_ptr()
        )

    def test_a_replica_whose_params_do_not_fit_the_shared_buffer_is_rejected(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """A param of another shape cannot alias the shared buffer without loading wrong bytes."""
        model_replicas = model_replicas_of_width({0: 4, 1: 3})
        model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))

        with pytest.raises(AssertionError, match="in the shared buffer"):
            model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))


def _copy_into_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def _reshape_then_copy(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data = param.data.view(loaded_weight.shape)
    param.data.copy_(loaded_weight)


def _parameter(*shape: int, weight_loader=_copy_into_param) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
    param.weight_loader = weight_loader
    return param


def _toy_model(model_replica_module: ModuleType, *, restores_expert_layout: bool = True) -> torch.nn.Module:
    """A model as sglang builds it, with the postprocess patterns the replica must reproduce: experts permuted into
    a block layout, a scale registered under a second name, and a param that only postprocess creates."""

    class _BlockExpertsMethod(model_replica_module.QuantizeMethodBase):
        def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            layer.w13.data = layer.w13.data.view(2, 2, 6)

        def restore_weights_before_loading(self, layer: torch.nn.Module) -> None:
            if restores_expert_layout:
                layer.w13.data = layer.w13.data.view(4, 6)

    class _SwizzledScaleMethod(model_replica_module.QuantizeMethodBase):
        def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            layer.register_parameter("weight_scale_swizzled", layer.weight_scale)
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.zeros(5, device=layer.weight.device), requires_grad=False
            )

    experts = torch.nn.Module()
    experts.w13 = _parameter(4, 6, weight_loader=_copy_into_param if restores_expert_layout else _reshape_then_copy)
    experts.quant_method = _BlockExpertsMethod()
    linear = torch.nn.Module()
    linear.weight = _parameter(3, 2)
    linear.weight_scale = _parameter(3)
    linear.quant_method = _SwizzledScaleMethod()
    norm = torch.nn.Module()
    norm.weight = _parameter(2)

    model = torch.nn.Module()
    model.experts, model.linear, model.norm = experts, linear, norm

    def load_weights(hf_tensors: list[tuple[str, torch.Tensor]]) -> None:
        model.server_args_seen_by_loads.append(model_replica_module.get_server_args())
        params_by_name = dict(model.named_parameters())
        for name, tensor in hf_tensors:
            params_by_name[name].weight_loader(params_by_name[name], tensor)

    model.load_weights = load_weights
    model.server_args_seen_by_loads = []
    return model


@pytest.fixture
def published_server_args(model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list:
    """sglang's one live config per process: the server args published last."""
    published = [_ServerArgs()]
    monkeypatch.setattr(model_replica_module, "get_server_args", lambda: published[-1])
    monkeypatch.setattr(model_replica_module, "_publish_server_args", published.append)
    return published


@pytest.fixture
def make_model_replica(model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch, published_server_args):
    """Builds a `ModelReplica` from `_toy_model` the way `build_model_replica` does: parameters swapped for 0-size
    ones, their built shapes passed along, postprocess on the CPU."""
    monkeypatch.setattr(model_replica_module, "ParallelismContext", lambda parallelism: nullcontext())
    monkeypatch.setattr(model_replica_module, "ParameterMapper", SimpleNamespace(from_model=lambda model: None))

    def make(server_args: _ServerArgs | None = None, **toy_model_kwargs):
        model = _toy_model(model_replica_module, **toy_model_kwargs)
        built_shapes_by_name = {name: param.shape for name, param in model.named_parameters()}
        for param in model.parameters():
            param.data = torch.empty(0, dtype=param.dtype)
        config = model_replica_module.RolloutEngineRankConfig(
            parallelism=None, server_args=server_args or _ServerArgs()
        )
        return model_replica_module.ModelReplica(
            model, built_shapes_by_name, config, postprocess_device=torch.device("cpu")
        )

    return make


def _float_tensor(start: float, *shape: int) -> torch.Tensor:
    numel = torch.Size(shape).numel()
    return torch.arange(start, start + numel, dtype=torch.float32).view(shape)


class TestModelReplica:
    def test_param_specs_are_the_state_the_engine_loads_into(self, make_model_replica) -> None:
        """The engine's loader writes into its params after postprocess and restore, so loads must see that state;
        an aliased param must keep its real size, and a param only postprocess creates is still published."""
        model_replica = make_model_replica()

        assert {name: (tuple(spec.shape), spec.nbytes) for name, spec in model_replica.param_specs.items()} == {
            "experts.w13": ((4, 6), 96),
            "linear.weight": ((3, 2), 24),
            "linear.weight_scale": ((3,), 12),
            "linear.weight_scale_inv": ((5,), 20),
            "norm.weight": ((2,), 8),
        }
        assert all(param.numel() == 0 for param in model_replica._model.parameters())

    def test_load_into_returns_the_loader_output_in_the_buffer(self, make_model_replica) -> None:
        """The returned bytes are what the transport writes, so they must be the loader's output and lie in the
        registered buffer."""
        model_replica = make_model_replica()
        buffer = torch.zeros(1024, dtype=torch.uint8)
        w13, norm = _float_tensor(1.0, 4, 6), _float_tensor(100.0, 2)

        spans_by_name = model_replica.load_into(
            buffer, ["experts.w13", "norm.weight"], [("experts.w13", w13), ("norm.weight", norm)]
        )

        assert torch.equal(spans_by_name["experts.w13"].view(torch.float32), w13.flatten())
        assert torch.equal(spans_by_name["norm.weight"].view(torch.float32), norm)
        for span in spans_by_name.values():
            assert buffer.data_ptr() <= span.data_ptr() < buffer.data_ptr() + buffer.numel()
        assert all(param.numel() == 0 for param in model_replica._model.parameters())

    def test_a_loader_that_reshapes_its_param_is_rejected(self, make_model_replica) -> None:
        """Without a restore the expert loader reshapes the block-layout param; a raw write would land canonical
        bytes on the engine's block-layout param."""
        model_replica = make_model_replica(restores_expert_layout=False)

        with pytest.raises(AssertionError, match="changed more than the bytes of experts.w13"):
            model_replica.load_into(
                torch.zeros(1024, dtype=torch.uint8), ["experts.w13"], [("experts.w13", _float_tensor(1.0, 4, 6))]
            )
        assert all(param.numel() == 0 for param in model_replica._model.parameters())

    def test_each_replica_loads_under_the_server_args_it_was_built_with(
        self, make_model_replica, published_server_args: list
    ) -> None:
        """Rollout engines launched with different sglang arguments get their own replicas in one process, while
        sglang holds one live config per process; a load must not run under another replica's."""
        trtllm_server_args = _ServerArgs(moe_runner_backend="flashinfer_trtllm")
        triton_server_args = _ServerArgs(moe_runner_backend="triton")
        trtllm_replica = make_model_replica(server_args=trtllm_server_args)
        triton_replica = make_model_replica(server_args=triton_server_args)

        for model_replica in (trtllm_replica, trtllm_replica, triton_replica, trtllm_replica):
            model_replica.load_into(
                torch.zeros(1024, dtype=torch.uint8), ["norm.weight"], [("norm.weight", _float_tensor(1.0, 2))]
            )

        assert trtllm_replica._model.server_args_seen_by_loads == [trtllm_server_args] * 3
        assert triton_replica._model.server_args_seen_by_loads == [triton_server_args]
        assert published_server_args[1:] == [trtllm_server_args, triton_server_args, trtllm_server_args]

    def test_hf_tensors_of_a_param_outside_the_group_fail_the_load(self, make_model_replica) -> None:
        """Only the group's params have storage, so an HF tensor of another param cannot be silently dropped."""
        model_replica = make_model_replica()

        with pytest.raises(RuntimeError):
            model_replica.load_into(
                torch.zeros(1024, dtype=torch.uint8),
                ["norm.weight"],
                [("norm.weight", _float_tensor(1.0, 2)), ("linear.weight", _float_tensor(1.0, 3, 2))],
            )


def test_param_spec_spans_every_byte_of_a_strided_layout(model_replica_module: ModuleType) -> None:
    """A strided param's span reaches its last element, so a buffer span of `nbytes` holds all of it."""
    columns = torch.empty(4, 6)[:, :3]

    assert model_replica_module.ParamSpec.of(columns).nbytes == (3 * 6 + 2 + 1) * 4
    assert model_replica_module.ParamSpec.of(torch.empty(4, 6).t()).nbytes == 24 * 4


class TestPackIntoBuffers:
    @staticmethod
    def _specs(model_replica_module: ModuleType, nbytes_by_name: dict[str, int]) -> dict:
        return {
            name: model_replica_module.ParamSpec.of(torch.empty(nbytes, dtype=torch.uint8))
            for name, nbytes in nbytes_by_name.items()
        }

    def test_every_group_fits_the_layout_load_into_gives_it(self, model_replica_module: ModuleType) -> None:
        """`load_into` aligns each param's span, so groups must be packed with the same alignment."""
        param_specs = self._specs(model_replica_module, {"a": 100, "b": 100, "c": 300, "d": 50})

        groups = list(model_replica_module.pack_into_buffers(["a", "b", "c", "d"], param_specs, buffer_bytes=512))

        assert [name for group in groups for name in group] == ["a", "b", "c", "d"]
        for group in groups:
            model_replica_module._spans_in_buffer(torch.empty(512, dtype=torch.uint8), group, param_specs)

    def test_a_param_larger_than_a_buffer_is_rejected(self, model_replica_module: ModuleType) -> None:
        param_specs = self._specs(model_replica_module, {"small": 100, "large": 600})

        with pytest.raises(AssertionError, match="large takes 600 bytes"):
            list(model_replica_module.pack_into_buffers(["small", "large"], param_specs, buffer_bytes=512))


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_drop_fields_this_sglang_does_not_know(
    model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch, record_factory
) -> None:
    """An engine on another sglang reports fields this ServerArgs lacks; they must not break the query."""
    server_args_type = record_factory("ServerArgs", [("model_path", str)])
    monkeypatch.setattr(model_replica_module, "ServerArgs", server_args_type)

    server_args = model_replica_module.create_server_args_from_dict({"model_path": "/model", "unknown_field": True})

    assert isinstance(server_args, server_args_type)
    assert server_args.model_path == "/model"
