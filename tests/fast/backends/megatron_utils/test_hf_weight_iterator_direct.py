import sys
import types
from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.types import ParamInfo


def _install_import_stubs(monkeypatch):
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.cdiv = lambda x, y: (x + y - 1) // y
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)

    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.utils.patch_torch",
        "sglang.srt.weight_sync",
        "sglang.srt.weight_sync.tensor_bucket",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp8_utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    sys.modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = object
    fp8_utils = sys.modules["sglang.srt.layers.quantization.fp8_utils"]
    fp8_utils.quant_weight_ue8m0 = lambda *args, **kwargs: None
    fp8_utils.transform_scale_ue8m0 = lambda x, **kwargs: x

    ray = types.ModuleType("ray")
    ray_actor = types.ModuleType("ray.actor")
    ray_util = types.ModuleType("ray.util")
    ray_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    ray.remote = lambda *args, **kwargs: args[0] if args and callable(args[0]) and not kwargs else lambda obj: obj
    ray_actor.ActorHandle = object
    ray_scheduling.NodeAffinitySchedulingStrategy = object
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_scheduling)

    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0


@pytest.fixture
def direct_module(monkeypatch):
    module_names = [
        "miles.backends.megatron_utils.sglang",
        "miles.backends.megatron_utils.megatron_to_hf",
        "miles.backends.megatron_utils.megatron_to_hf.processors",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8",
        "miles.backends.megatron_utils.named_weights",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    _install_import_stubs(monkeypatch)

    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct

    yield hf_weight_iterator_direct

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _param(name: str, size: int) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=0,
    )


def test_gather_batches_pack_by_size_only(direct_module, monkeypatch):
    # Atomicity is the base template's job; gather batches are pure size packing.
    params = [_param("layer.a", 4), _param("layer.b", 2), _param("layer.c", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)

    batches = direct_module._pack_param_infos_by_size(Namespace(update_weight_buffer_size=6), params)
    assert [[param.name for param in batch] for batch in batches] == [["layer.a", "layer.b"], ["layer.c"]]

    batches = direct_module._pack_param_infos_by_size(
        Namespace(update_weight_buffer_size=6), params, size_multiplier=2
    )
    assert [[param.name for param in batch] for batch in batches] == [["layer.a"], ["layer.b"], ["layer.c"]]


@pytest.fixture
def named_weights_module(monkeypatch):
    _install_import_stubs(monkeypatch)
    from miles.backends.megatron_utils import named_weights

    return named_weights


@pytest.mark.parametrize("ep_rank", [0, 1, 3])
@pytest.mark.parametrize(
    "prefix", ["decoder.layers.7", "mtp.layers.0.transformer_layer", "mtp.layers.1.mtp_model_layer"]
)
@pytest.mark.parametrize("fc,partition_dim", [("linear_fc1", 0), ("linear_fc2", 1)])
@pytest.mark.parametrize("native_storage", [False, True])
def test_unpack_grouped_expert_names_values_and_partition_attrs(
    named_weights_module, monkeypatch, ep_rank, prefix, fc, partition_dim, native_storage
):
    monkeypatch.setattr(
        named_weights_module, "get_parallel_state", lambda: Namespace(ep=Namespace(rank=ep_rank, size=4))
    )
    param = torch.nn.Parameter(torch.arange(48, dtype=torch.float32).view(2, 6, 4))
    if native_storage:
        param.rowwise_data = param.detach().flatten()
        param.quantizer = None
    param.tensor_model_parallel = True
    param.partition_dim = partition_dim
    param.partition_stride = 1
    param.parallel_mode = None
    name = f"module.module.{prefix}.mlp.experts.{fc}.weight"

    actual = list(named_weights_module.unpack_grouped_expert_weights(Namespace(num_experts=8), [(name, param)]))

    assert [n for n, _ in actual] == [f"{name}{2 * ep_rank}", f"{name}{2 * ep_rank + 1}"]
    for i, (_, weight) in enumerate(actual):
        torch.testing.assert_close(weight, param[i], rtol=0, atol=0)
        assert weight.data_ptr() == param[i].data_ptr()
        assert weight.tensor_model_parallel
        assert weight.partition_dim == partition_dim
        assert weight.partition_stride == 1
        assert weight.parallel_mode is None
    with torch.no_grad():
        param.add_(100)
    torch.testing.assert_close(actual[1][1], param[1], rtol=0, atol=0)


def test_unpack_grouped_experts_preserves_discrete_and_shared_weights(named_weights_module, monkeypatch):
    monkeypatch.setattr(named_weights_module, "get_parallel_state", lambda: Namespace(ep=Namespace(rank=1, size=2)))
    weight = torch.nn.Parameter(torch.ones(4, 8))
    names = [
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight4",
        "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
        "module.module.decoder.layers.0.mlp.router.weight",
    ]
    actual = list(
        named_weights_module.unpack_grouped_expert_weights(Namespace(num_experts=8), [(n, weight) for n in names])
    )
    assert [n for n, _ in actual] == names
    assert all(p is weight for _, p in actual)


def test_unpack_grouped_experts_rejects_wrong_expert_count(named_weights_module, monkeypatch):
    monkeypatch.setattr(named_weights_module, "get_parallel_state", lambda: Namespace(ep=Namespace(rank=0, size=2)))
    name = "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight"
    with pytest.raises(ValueError, match="Expected 4 packed expert weights"):
        list(
            named_weights_module.unpack_grouped_expert_weights(Namespace(num_experts=8), [(name, torch.ones(2, 4, 8))])
        )


def test_grouped_cpu_backup_reads_storage_before_expert_views(named_weights_module, monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, 4, 8))
    param.rowwise_data = param.detach().flatten()
    backup = torch.arange(64, dtype=torch.float32)
    calls = []

    def get_cpu_backup(tensor, *, zero_copy):
        assert zero_copy
        calls.append(tensor)
        return backup

    monkeypatch.setitem(
        sys.modules, "torch_memory_saver", Namespace(torch_memory_saver=Namespace(get_cpu_backup=get_cpu_backup))
    )
    result = named_weights_module._maybe_get_cpu_backup(param)

    assert calls[0] is param.rowwise_data
    assert result.shape == param.shape
    assert result.data_ptr() == backup.data_ptr()
    torch.testing.assert_close(result, backup.view_as(param), rtol=0, atol=0)
