import sys
import types
from argparse import Namespace
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
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
    sys.modules["megatron.core.utils"].unwrap_model = lambda model: model
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


def _param(name: str, size: int, *, dtype=torch.float32) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=dtype,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=0,
    )


def _make_direct_iterator(
    direct_module,
    monkeypatch,
    *,
    expert_infos,
    model_name="qwen3moe",
    quantization_config=None,
    placement=None,
    peer_capabilities=None,
    ep_size=4,
):
    placement = placement or WeightUpdatePlacement(gather_pp=False, gather_ep=False)

    def fake_super_init(self, args, model, *, placement, model_name, quantization_config):
        self.args = args
        self.model = model
        self.placement = placement
        self.model_name = model_name
        self.quantization_config = quantization_config

    monkeypatch.setattr(direct_module.MegatronHfWeightIteratorBase, "__init__", fake_super_init)
    monkeypatch.setattr(
        direct_module,
        "_get_megatron_local_param_infos",
        lambda *args, **kwargs: ([], expert_infos),
    )
    monkeypatch.setattr(
        direct_module,
        "get_parallel_state",
        lambda: SimpleNamespace(ep=SimpleNamespace(size=ep_size)),
    )
    monkeypatch.setattr(direct_module, "get_gloo_group", lambda: "gloo")

    world_size = len(peer_capabilities) if peer_capabilities is not None else 2
    monkeypatch.setattr(direct_module.dist, "get_world_size", lambda *args, **kwargs: world_size)

    def fake_all_gather_object(object_list, obj, group=None):
        assert group == "gloo"
        object_list[:] = peer_capabilities if peer_capabilities is not None else [obj] * world_size

    monkeypatch.setattr(direct_module.dist, "all_gather_object", fake_all_gather_object)

    expert_batch_multipliers = []

    def fake_pack(_args, infos, *, size_multiplier=1):
        if infos:
            expert_batch_multipliers.append(size_multiplier)
        return [list(infos)] if infos else []

    monkeypatch.setattr(direct_module, "_pack_param_infos_by_size", fake_pack)
    iterator = direct_module.HfWeightIteratorDirect(
        Namespace(update_weight_buffer_size=1024),
        [],
        placement=placement,
        model_name=model_name,
        quantization_config=quantization_config,
    )
    return iterator, expert_batch_multipliers


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


def test_direct_iterator_can_keep_pp_and_ep_local(direct_module):
    assert direct_module.HfWeightIteratorDirect.forced_placement == WeightUpdatePlacement(
        gather_pp=False,
        gather_ep=False,
    )


def test_individual_bf16_qwen3moe_experts_keep_ep_local_and_pack_local_size(direct_module, monkeypatch):
    expert = _param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight7",
        8,
        dtype=torch.bfloat16,
    )

    iterator, multipliers = _make_direct_iterator(direct_module, monkeypatch, expert_infos=[expert])

    assert iterator.placement.gather_ep is False
    assert multipliers == [1]


@pytest.mark.parametrize(
    ("model_name", "quantization_config", "expert_name", "dtype"),
    [
        ("deepseekv3", None, "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight7", torch.bfloat16),
        ("qwen3moe", {}, "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight7", torch.bfloat16),
        ("qwen3moe", None, "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight", torch.bfloat16),
        ("qwen3moe", None, "module.module.decoder.layers.0.mlp.experts.other.weight7", torch.bfloat16),
        ("qwen3moe", None, "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight7", torch.float32),
    ],
)
def test_unsupported_expert_export_falls_back_to_ep_gather(
    direct_module,
    monkeypatch,
    model_name,
    quantization_config,
    expert_name,
    dtype,
):
    expert = _param(expert_name, 8, dtype=dtype)

    iterator, multipliers = _make_direct_iterator(
        direct_module,
        monkeypatch,
        expert_infos=[expert],
        model_name=model_name,
        quantization_config=quantization_config,
    )

    assert iterator.placement.gather_ep is True
    assert multipliers == [4]


def test_one_unsupported_rank_makes_every_rank_fall_back_to_ep_gather(direct_module, monkeypatch):
    expert = _param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight3",
        8,
        dtype=torch.bfloat16,
    )

    iterator, multipliers = _make_direct_iterator(
        direct_module,
        monkeypatch,
        expert_infos=[expert],
        peer_capabilities=[(True, True), (True, False)],
    )

    assert iterator.placement.gather_ep is True
    assert multipliers == [4]


def test_a_rank_without_expert_metadata_makes_every_rank_fall_back_to_ep_gather(direct_module, monkeypatch):
    expert = _param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight3",
        8,
        dtype=torch.bfloat16,
    )

    iterator, multipliers = _make_direct_iterator(
        direct_module,
        monkeypatch,
        expert_infos=[expert],
        peer_capabilities=[(True, True), (False, True)],
    )

    assert iterator.placement.gather_ep is True
    assert multipliers == [4]


def test_single_ep_rank_keeps_the_legacy_gathered_placement(direct_module, monkeypatch):
    expert = _param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        8,
        dtype=torch.bfloat16,
    )

    iterator, multipliers = _make_direct_iterator(
        direct_module,
        monkeypatch,
        expert_infos=[expert],
        ep_size=1,
    )

    assert iterator.placement.gather_ep is True
    assert multipliers == [1]


def test_materialize_expert_batch_without_ep_gather_still_gathers_etp(direct_module, monkeypatch):
    info = _param(
        "module.module.decoder.layers.0.mlp.experts.linear_fc2.weight3",
        2,
        dtype=torch.bfloat16,
    )
    local = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    etp_gathered = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    etp_calls = []

    monkeypatch.setattr(direct_module, "monkey_patch_torch_reductions", lambda: None)
    monkeypatch.setattr(direct_module, "_load_or_allocate_params", lambda *args: [local])
    monkeypatch.setattr(direct_module, "_set_tp_attrs", lambda *args: None)

    def fake_gather_etp(args, infos_and_params):
        etp_calls.append(infos_and_params)
        return [etp_gathered]

    monkeypatch.setattr(direct_module, "all_gather_params_async", fake_gather_etp)
    monkeypatch.setattr(
        direct_module,
        "get_parallel_state",
        lambda: SimpleNamespace(ep=SimpleNamespace(size=4, group="ep")),
    )

    def unexpected_ep_collective(*args, **kwargs):
        raise AssertionError("EP collectives must not run for an EP-local expert batch")

    monkeypatch.setattr(direct_module.dist, "all_gather_object", unexpected_ep_collective)
    monkeypatch.setattr(direct_module.dist, "all_gather", unexpected_ep_collective)

    result = direct_module._materialize_expert_batch(
        Namespace(),
        [info],
        {},
        gather_pp=False,
        gather_ep=False,
    )

    assert len(etp_calls) == 1
    assert [name for name, _tensor in result] == [info.name]
    assert result[0][1] is etp_gathered
