import dataclasses
from contextlib import contextmanager, nullcontext
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
        """`ModelReplicas` whose replica for tp rank r lays out one `weight` of `widths[r]` floats and loads it from
        `hf_names_by_rank[r]`."""

        def make(widths: dict[int, int], hf_names_by_rank: dict[int, set[str]] | None = None):
            def build_model_replica(config, model_path, *, transfer_buffer_device):
                tp_rank = config.parallelism.tp_rank
                param_layout = model_replica_module.TransferBufferParamLayout.from_tensor(torch.empty(widths[tp_rank]))

                def map_hf_names(hf_tensor_specs):
                    mapping_calls.append(tp_rank)
                    return model_replica_module.HfNameMapping.from_hf_names_by_param_name(
                        {"weight": frozenset((hf_names_by_rank or {}).get(tp_rank, ()))}
                    )

                return SimpleNamespace(
                    transfer_buffer_param_layouts={"weight": param_layout}, map_hf_names=map_hf_names
                )

            mapping_calls = []

            monkeypatch.setattr(model_replica_module, "build_model_replica", build_model_replica)
            model_replicas = model_replica_module.ModelReplicas(
                model_path="/model", transfer_buffer_device=torch.device("cpu")
            )
            model_replicas.mapping_calls = mapping_calls
            return model_replicas

        return make

    def test_each_shard_layout_gets_one_replica_for_the_process(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """Rollout engines holding a rank the same way share its replica, so a reconnect or another engine builds
        nothing new."""
        model_replicas = model_replicas_of_width({0: 4, 1: 4})

        first_engine_rank_0 = model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))
        second_engine_rank_0 = model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=8))
        rank_1 = model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))

        assert second_engine_rank_0 is first_engine_rank_0
        assert rank_1 is not first_engine_rank_0

    def test_a_replica_laid_out_differently_from_the_others_is_rejected(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """One packing of ready params serves every rank a sender writes, so their layouts must match."""
        model_replicas = model_replicas_of_width({0: 4, 1: 3})
        model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))

        with pytest.raises(AssertionError, match="lays out weight"):
            model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))

    def test_a_param_waits_for_the_hf_names_any_replica_loads_into_it(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """With experts split across ranks each replica loads only its own, but one packing serves them all."""
        model_replicas = model_replicas_of_width({0: 4, 1: 4}, hf_names_by_rank={0: {"expert.0"}, 1: {"expert.1"}})
        model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))
        model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))

        model_replicas.map_hf_names({})
        model_replicas.map_hf_names({})

        assert model_replicas.hf_name_mapping.hf_names_by_param_name == {"weight": {"expert.0", "expert.1"}}
        assert model_replicas.mapping_calls == [0, 1]


def _copy_into_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def _reshape_then_copy(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data = param.data.view(loaded_weight.shape)
    param.data.copy_(loaded_weight)


def _parameter(*shape: int, weight_loader=_copy_into_param) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
    param.weight_loader = weight_loader
    return param


def _toy_model(
    model_replica_module: ModuleType, *, restores_expert_layout: bool = True, derives_norm_buffer: bool = False
) -> torch.nn.Module:
    """A model as sglang builds it, with the postprocess patterns the replica must reproduce: experts permuted into
    a block layout, a scale registered under a second name, and a param that only postprocess creates. With
    `derives_norm_buffer`, the norm's loader also writes `weight + 1` into a buffer, as sglang's Gemma norm does."""

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
    if derives_norm_buffer:
        # buffers stay on meta, as the replica builds them
        norm.register_buffer("weight_plus_one", torch.empty(2, device="meta"), persistent=False)
        norm.register_buffer("cos_sin_cache", torch.empty(3, device="meta"), persistent=False)
        norm.weight.weight_loader = lambda param, loaded_weight: (
            param.data.copy_(loaded_weight),
            torch.add(param.data, 1.0, out=norm.weight_plus_one),
        )

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

    def make(server_args: _ServerArgs | None = None, **toy_model_kwargs):
        model = _toy_model(model_replica_module, **toy_model_kwargs)
        built_shapes_by_name = {name: param.shape for name, param in model.named_parameters()}
        for param in model.parameters():
            param.data = torch.empty(0, dtype=param.dtype)
        config = model_replica_module.RolloutEngineRankConfig(
            parallelism=None, server_args=server_args or _ServerArgs()
        )
        return model_replica_module.ModelReplica(
            model,
            built_shapes_by_name,
            config,
            postprocess_device=torch.device("cpu"),
            transfer_buffer_device=torch.device("cpu"),
        )

    return make


def _float_tensor(start: float, *shape: int) -> torch.Tensor:
    numel = torch.Size(shape).numel()
    return torch.arange(start, start + numel, dtype=torch.float32).view(shape)


class TestModelReplica:
    def test_param_layouts_are_the_state_the_engine_loads_into(self, make_model_replica) -> None:
        """The engine's loader writes into its params after postprocess and restore, so loads must see that state;
        an aliased param must keep its real size, and a param only postprocess creates is still published."""
        model_replica = make_model_replica()

        assert {
            name: (tuple(param_layout.shape), param_layout.occupied_nbytes)
            for name, param_layout in model_replica.transfer_buffer_param_layouts.items()
        } == {
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

        param_bytes_by_name = model_replica.load_into(
            buffer, ["experts.w13", "norm.weight"], [("experts.w13", w13), ("norm.weight", norm)]
        )

        assert torch.equal(param_bytes_by_name["experts.w13"].view(torch.float32), w13.flatten())
        assert torch.equal(param_bytes_by_name["norm.weight"].view(torch.float32), norm)
        for param_bytes in param_bytes_by_name.values():
            assert buffer.data_ptr() <= param_bytes.data_ptr() < buffer.data_ptr() + buffer.numel()
        assert all(param.numel() == 0 for param in model_replica._model.parameters())

    def test_load_into_runs_inside_the_rank_parallelism_context(
        self, model_replica_module: ModuleType, make_model_replica, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sglang's sharded weight loaders read the rank at call time, which exists only inside its context."""
        model_replica = make_model_replica()
        events = []

        @contextmanager
        def logged_parallelism_context(parallelism):
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr(model_replica_module, "ParallelismContext", logged_parallelism_context)
        load_weights = model_replica._model.load_weights
        model_replica._model.load_weights = lambda hf_tensors: (events.append("load"), load_weights(hf_tensors))

        model_replica.load_into(
            torch.zeros(1024, dtype=torch.uint8), ["norm.weight"], [("norm.weight", _float_tensor(1.0, 2))]
        )

        assert events == ["enter", "load", "exit"]

    def test_map_hf_names_runs_the_replica_loader_inside_the_rank_parallelism_context(
        self, model_replica_module: ModuleType, make_model_replica, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mapping comes from the loader that `load_into` runs, under the same rank."""
        model_replica = make_model_replica()
        events = []

        @contextmanager
        def logged_parallelism_context(parallelism):
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr(model_replica_module, "ParallelismContext", logged_parallelism_context)
        load_weights = model_replica._model.load_weights
        model_replica._model.load_weights = lambda hf_tensors: (events.append("load"), load_weights(hf_tensors))
        layouts = model_replica.transfer_buffer_param_layouts
        hf_tensor_specs = {
            name: model_replica_module.HfTensorSpec(tuple(layouts[name].shape), layouts[name].dtype)
            for name in ("experts.w13", "norm.weight")
        }

        hf_name_mapping = model_replica.map_hf_names(hf_tensor_specs)

        assert hf_name_mapping.hf_names_by_param_name == {
            "experts.w13": {"experts.w13"},
            "norm.weight": {"norm.weight"},
        }
        assert events == ["enter", "load", "exit"]
        assert all(param.numel() == 0 for param in model_replica._model.parameters())

    def test_buffers_the_loader_writes_get_storage_once_names_are_mapped(
        self, model_replica_module: ModuleType, make_model_replica
    ) -> None:
        """sglang's Gemma norm loader also writes `weight + 1` into a buffer the replica builds without storage; a
        load would fail there. Buffers the loader never writes stay without storage."""
        model_replica = make_model_replica(derives_norm_buffer=True)
        model_replica.map_hf_names({"norm.weight": model_replica_module.HfTensorSpec((2,), torch.float32)})

        param_bytes_by_name = model_replica.load_into(
            torch.zeros(1024, dtype=torch.uint8), ["norm.weight"], [("norm.weight", _float_tensor(1.0, 2))]
        )

        assert torch.equal(param_bytes_by_name["norm.weight"].view(torch.float32), _float_tensor(1.0, 2))
        assert model_replica._model.norm.weight_plus_one.device.type == "cpu"
        assert model_replica._model.norm.cos_sin_cache.is_meta

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


def test_a_strided_param_occupies_the_bytes_up_to_its_last_element(model_replica_module: ModuleType) -> None:
    """A param whose strides leave gaps needs the bytes up to its last element, so its view fits in the buffer."""
    first_three_columns = torch.empty(4, 6)[:, :3]

    assert (
        model_replica_module.TransferBufferParamLayout.from_tensor(first_three_columns).occupied_nbytes
        == (3 * 6 + 2 + 1) * 4
    )
    assert model_replica_module.TransferBufferParamLayout.from_tensor(torch.empty(4, 6).t()).occupied_nbytes == 24 * 4


class TestPackIntoBuffers:
    @staticmethod
    def _layouts(model_replica_module: ModuleType, nbytes_by_name: dict[str, int]) -> dict:
        return {
            name: model_replica_module.TransferBufferParamLayout.from_tensor(torch.empty(nbytes, dtype=torch.uint8))
            for name, nbytes in nbytes_by_name.items()
        }

    def test_every_pack_fits_the_layout_load_into_gives_it(self, model_replica_module: ModuleType) -> None:
        """`load_into` aligns where each param starts, so packs must be built with the same alignment."""
        param_layouts = self._layouts(model_replica_module, {"a": 100, "b": 100, "c": 300, "d": 50})

        packs = list(
            model_replica_module.pack_into_buffers([("a",), ("b",), ("c",), ("d",)], param_layouts, buffer_nbytes=512)
        )

        assert [name for pack in packs for group in pack for name in group] == ["a", "b", "c", "d"]
        for pack in packs:
            param_names = [name for group in pack for name in group]
            model_replica_module._slice_buffer_by_param(
                torch.empty(512, dtype=torch.uint8), param_names, param_layouts
            )

    def test_the_params_of_a_group_stay_in_one_pack(self, model_replica_module: ModuleType) -> None:
        """They load from shared HF tensors, so splitting them would hand the loader a param that is not bound."""
        param_layouts = self._layouts(model_replica_module, {"a": 200, "b": 200, "c": 200})

        packs = list(model_replica_module.pack_into_buffers([("a",), ("b", "c")], param_layouts, buffer_nbytes=512))

        assert packs == [[("a",)], [("b", "c")]]

    def test_a_group_larger_than_a_buffer_is_rejected(self, model_replica_module: ModuleType) -> None:
        """No buffer can hold it, so packing must fail rather than hand out a pack that overflows."""
        param_layouts = self._layouts(model_replica_module, {"small": 100, "large": 600})

        with pytest.raises(AssertionError, match=r"\('large',\) need 600 bytes"):
            list(model_replica_module.pack_into_buffers([("small",), ("large",)], param_layouts, buffer_nbytes=512))


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
