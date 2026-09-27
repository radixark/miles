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
        "megatron.core.utils",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0
    sys.modules["megatron.core.utils"].unwrap_model = lambda model: model


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


def _param(name: str, size: int, *, src_rank: int = 0) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=src_rank,
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


@pytest.mark.parametrize("materialize", [True, False])
@pytest.mark.parametrize("consume_local", [True, False])
def test_expert_consumer_bypasses_gather_on_every_owner(direct_module, monkeypatch, materialize, consume_local):
    """The owner consumes any converted layout; other transports keep the gathered stream."""
    local_name = "layer.experts.linear_fc1.weight0"
    local, remote = _param(local_name, 4), _param("layer.experts.linear_fc1.weight1", 4, src_rank=1)
    unit = [
        ("bf16.weight", torch.ones(4, dtype=torch.bfloat16)),
        ("fp8.weight", torch.ones(4, dtype=torch.float8_e4m3fn)),
        ("packed.weight", torch.ones(4, dtype=torch.uint8)),
        ("scale", torch.ones(())),
    ]
    events = []

    class Weight:
        def detach(self):
            return self

        def to(self, **kwargs):
            events.append("load")
            return unit[0][1]

    def convert(named_params):
        assert [name for name, _ in named_params] == [local_name]
        events.append("convert")
        yield unit

    def consume(converted):
        assert converted is unit
        events.append("consume")

    def gather(units, **kwargs):
        assert not consume_local
        assert units == [unit]
        events.append("gather")
        return units

    iterator = direct_module.HfWeightIteratorDirect.__new__(direct_module.HfWeightIteratorDirect)
    iterator.args = Namespace()
    iterator._non_expert_batches = []
    iterator._expert_batches = [direct_module._ExpertBatch(param_infos=[local, remote], gathers=(gather,))]
    iterator._convert_to_hf_param_units = convert
    iterator._local_expert_consumer = None
    if consume_local:
        iterator.set_local_expert_consumer(consume)
    monkeypatch.setattr(direct_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(direct_module, "_iter_mm_tower_units", lambda *args, **kwargs: iter(()))

    units = list(iterator._iter_hf_param_units({local_name: Weight()}, materialize=materialize))
    assert units == ([unit] if materialize and not consume_local else [])
    assert events == ["load", "convert", "consume" if consume_local else "gather"]
