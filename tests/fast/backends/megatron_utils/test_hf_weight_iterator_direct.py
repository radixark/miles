import sys
import types
from argparse import Namespace
from types import SimpleNamespace

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
    fp8_utils.mxfp8_group_quantize = lambda *args, **kwargs: None
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
        "miles.backends.megatron_utils.update_weight.common",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin",
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


def test_atomic_group_is_single_update_unit_and_packed_together(direct_module, monkeypatch):
    from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup

    params = [_param("layer.a", 4), _param("layer.b", 4), _param("layer.c", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)

    update_units = direct_module.get_named_update_units(
        [param.name for param in params], [AtomicUpdateGroup("pair", (".b", ".c"))]
    )
    assert [unit.names for unit in update_units] == [("layer.a",), ("layer.b", "layer.c")]

    buckets = direct_module._pack_update_units(Namespace(update_weight_buffer_size=6), params, update_units)
    assert [[param.name for param in bucket] for bucket in buckets] == [["layer.a"], ["layer.b", "layer.c"]]


def test_deepseekv4_atomic_groups_use_named_update_units(direct_module):
    from miles.backends.megatron_utils.update_weight.common import get_atomic_update_groups

    param_names = [
        "module.module.decoder.layers.0.input_layernorm.weight",
        "module.module.decoder.layers.0.self_attention.wq_a.weight",
        "module.module.decoder.layers.0.self_attention.wkv.weight",
        "module.module.decoder.layers.0.self_attention.compressor.wkv.weight",
        "module.module.decoder.layers.0.self_attention.compressor.wgate.weight",
        "module.module.decoder.layers.0.self_attention.indexer.compressor.wkv.weight",
        "module.module.decoder.layers.0.self_attention.indexer.compressor.wgate.weight",
    ]

    update_units = direct_module.get_named_update_units(
        param_names, get_atomic_update_groups(Namespace(q_lora_rank=1024), "deepseekv4")
    )

    assert [unit.names for unit in update_units] == [
        ("module.module.decoder.layers.0.input_layernorm.weight",),
        (
            "module.module.decoder.layers.0.self_attention.wq_a.weight",
            "module.module.decoder.layers.0.self_attention.wkv.weight",
        ),
        (
            "module.module.decoder.layers.0.self_attention.compressor.wkv.weight",
            "module.module.decoder.layers.0.self_attention.compressor.wgate.weight",
        ),
        (
            "module.module.decoder.layers.0.self_attention.indexer.compressor.wkv.weight",
            "module.module.decoder.layers.0.self_attention.indexer.compressor.wgate.weight",
        ),
    ]


def test_atomic_group_specs_raise_explicit_errors(direct_module, monkeypatch):
    from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup

    params = [_param("layer.a", 4), _param("layer.b", 4)]

    invalid_groups = [
        ([AtomicUpdateGroup("empty", ())], "Atomic update group empty has no suffixes"),
        ([AtomicUpdateGroup("missing", (".c",))], "Atomic update group missing references no params"),
        (
            [AtomicUpdateGroup("left", (".a",)), AtomicUpdateGroup("right", (".a",))],
            "Param layer.a matches multiple atomic update groups",
        ),
        (
            [AtomicUpdateGroup("duplicate", (".a",)), AtomicUpdateGroup("duplicate", (".b",))],
            "Duplicate atomic update group: duplicate",
        ),
    ]

    for groups, error in invalid_groups:
        with pytest.raises(AssertionError, match=error):
            direct_module.get_named_update_units([param.name for param in params], groups)


def _tensor(size: int) -> torch.Tensor:
    return torch.empty(size, dtype=torch.uint8)


def _distributed_updater(mixin_module):
    updater = mixin_module.DistBucketedWeightUpdateMixin()
    updater.args = Namespace(update_weight_buffer_size=6)
    updater.model = []
    updater.model_name = "test-model"
    updater.quantization_config = None
    updater._is_source = True
    return updater


def test_distributed_non_expert_update_units_are_packed_together(direct_module, monkeypatch):
    from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import mixin

    updater = _distributed_updater(mixin)
    named_tensors = [("a", _tensor(4)), ("b", _tensor(4)), ("c", _tensor(4))]
    monkeypatch.setattr(mixin.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        mixin, "collect_named_tensors_for_weight_transfer", lambda *args, **kwargs: iter(named_tensors)
    )
    monkeypatch.setattr(
        mixin, "get_atomic_update_groups", lambda args, model_name: [AtomicUpdateGroup("pair", ("b", "c"))]
    )
    monkeypatch.setattr(mixin, "all_gather_param", lambda args, name, param: param)
    monkeypatch.setattr(
        mixin, "convert_to_hf", lambda args, model_name, name, param, quantization_config: [(name, param)]
    )

    buckets = []
    updater._gather_and_update_non_expert_weights(lambda tensors, pbar: buckets.append([name for name, _ in tensors]))

    assert buckets == [["a"], ["b", "c"]]


def test_distributed_expert_update_units_are_packed_together(direct_module, monkeypatch):
    from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import mixin

    updater = _distributed_updater(mixin)
    named_tensors = [
        ("module.experts.a", _tensor(4)),
        ("module.experts.b", _tensor(4)),
        ("module.experts.c", _tensor(4)),
    ]
    monkeypatch.setattr(mixin.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        mixin, "collect_named_tensors_for_weight_transfer", lambda *args, **kwargs: iter(named_tensors)
    )
    monkeypatch.setattr(
        mixin,
        "get_atomic_update_groups",
        lambda args, model_name: [AtomicUpdateGroup("pair", (".b", ".c"))],
    )
    monkeypatch.setattr(mixin, "all_gather_param", lambda args, name, param: param)
    monkeypatch.setattr(mixin, "get_parallel_state", lambda: SimpleNamespace(ep=SimpleNamespace(size=1)))

    buckets = []
    updater._update_expert_bucket_weights = lambda tensors, update_func, pbar: buckets.append(
        [name for name, _ in tensors]
    )

    updater._gather_and_update_expert_weights(lambda tensors, pbar: None)

    assert buckets == [["module.experts.a"], ["module.experts.b", "module.experts.c"]]


def test_distributed_atomic_group_cannot_span_expert_and_non_expert(direct_module, monkeypatch):
    from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import mixin

    updater = _distributed_updater(mixin)
    named_tensors = [("module.a", _tensor(4)), ("module.experts.b", _tensor(4))]
    monkeypatch.setattr(mixin.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        mixin, "collect_named_tensors_for_weight_transfer", lambda *args, **kwargs: iter(named_tensors)
    )
    monkeypatch.setattr(
        mixin,
        "get_atomic_update_groups",
        lambda args, model_name: [AtomicUpdateGroup("mixed", (".a", ".experts.b"))],
    )

    with pytest.raises(AssertionError, match="module.a"):
        updater._get_weight_transfer_update_units(is_expert=False)
