from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.target_modules import (
    resolve_megatron_lora_targets,
    select_present_target_modules,
    validate_lora_target_adapters,
)
from miles.utils.hf_lora_targets import resolve_hf_lora_targets


def _mapping(megatron, *hf):
    return SimpleNamespace(megatron_param=megatron, hf_param=hf[0] if len(hf) == 1 else dict(enumerate(hf)))


def _model(*names):
    model = torch.nn.Module()
    for name in names:
        module = model
        parts = name.split(".")
        for part in parts[:-1]:
            if part not in module._modules:
                module.add_module(part, torch.nn.Module())
            module = module._modules[part]
        module.add_module(parts[-1], torch.nn.Linear(4, 4, bias=False))
    return model


_QKV = _mapping(
    "decoder.layers.*.self_attention.linear_qkv.weight",
    *(f"model.layers.*.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
)


def test_scoped_attention_excludes_mtp():
    targets = resolve_hf_lora_targets({"model_type": "qwen3"}, train_attn=True, train_mlp=False, train_unembed=False)
    output = _mapping("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight")
    mtp = _mapping("mtp.layers.*.self_attention.linear_proj.weight", "mtp.layers.*.self_attn.o_proj.weight")
    candidates = resolve_megatron_lora_targets(targets, [_QKV, output, mtp], canonical=False)
    model = _model(
        "decoder.layers.0.self_attention.linear_qkv",
        "decoder.layers.0.self_attention.linear_proj",
        "mtp.layers.0.self_attention.linear_proj",
    )
    selected = select_present_target_modules([model], candidates)
    assert set(selected) == {
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.linear_proj",
    }
    model.requires_grad_(False)
    model.decoder.layers.get_submodule("0").self_attention.linear_qkv.weight.requires_grad_(True)
    with pytest.raises(AssertionError, match="LoRA injection skipped.*linear_proj"):
        validate_lora_target_adapters([model], selected)
    model.decoder.layers.get_submodule("0").self_attention.linear_proj.weight.requires_grad_(True)
    validate_lora_target_adapters([model], selected)
    assert not model.mtp.layers.get_submodule("0").self_attention.linear_proj.weight.requires_grad


def test_fused_selection_cannot_silently_expand():
    targets = ["model.layers.*.self_attn.q_proj"]
    with pytest.raises(AssertionError, match="requires all HF targets"):
        resolve_megatron_lora_targets(targets, [_QKV], canonical=False)
    candidates = resolve_megatron_lora_targets(targets, [_QKV], canonical=True)
    model = _model("decoder.layers.0.self_attention.linear_qkv")
    assert list(select_present_target_modules([model], candidates)) == ["decoder.layers.*.self_attention.linear_q"]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "sequential"])
def test_expert_representations_are_alternatives(grouped):
    target = "model.layers.*.mlp.experts.*.down_proj"
    mappings = [
        _mapping("decoder.layers.*.mlp.experts.linear_fc2.weight*", target + ".weight"),
        _mapping("decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight", target + ".weight"),
    ]
    candidates = resolve_megatron_lora_targets([target], mappings, canonical=False)
    module = "decoder.layers.0.mlp.experts." + ("linear_fc2" if grouped else "local_experts.0.linear_fc2")
    selected = select_present_target_modules([_model(module)], candidates)
    assert len(selected) == 1
    assert ("local_experts" in next(iter(selected))) != grouped
    with pytest.raises(AssertionError, match="no Megatron modules"):
        select_present_target_modules([_model("decoder.layers.0.mlp.linear_fc2")], candidates)


def test_missing_hf_mapping_is_not_a_megatron_passthrough():
    with pytest.raises(AssertionError, match="no Bridge mapping"):
        resolve_megatron_lora_targets(["model.layers.*.self_attn.unknown_proj"], [_QKV], canonical=False)


def test_one_to_one_mapping_keeps_bridge_module_name():
    target = "model.layers.*.self_attn.o_proj"
    mapping = _mapping("decoder.layers.*.self_attention.output_projection.weight", target + ".weight")
    candidates = resolve_megatron_lora_targets([target], [mapping], canonical=True)
    assert list(candidates) == ["decoder.layers.*.self_attention.output_projection"]
