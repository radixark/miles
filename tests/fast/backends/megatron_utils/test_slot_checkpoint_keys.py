"""A slot checkpoint's storage keys are slot-agnostic, so any free slot can reload it."""

from types import SimpleNamespace

import torch
from safetensors import safe_open

from miles.backends.megatron_utils.lora.checkpoint import DistributedOptimizerMembers, _canonicalize_slot_keys
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import write_peft_export
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets


def test_saved_and_loading_slots_meet_at_the_canonical_key():
    saved = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.2.linear_in.weight")
    loading = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.0.linear_in.weight")
    _canonicalize_slot_keys({"weights": {"w": saved}}, 2)
    _canonicalize_slot_keys({"weights": {"w": loading}}, 0)
    assert saved.key == loading.key == "decoder.layers.3.linear_qkv.adapter.linear_in.weight"


def test_single_lora_peft_export_names_native_experts_by_hf_module(tmp_path):
    """Native's single-LoRA export stacks experts under sglang's w1/w3/w2; --save's PEFT export carries HF names."""
    targets = get_hf_lora_targets({"model_type": "qwen3_moe", "num_hidden_layers": 1, "num_experts": 2})
    experts = "model.layers.0.mlp.experts"
    native_export = [
        (f"{experts}.w1.lora_A.weight", torch.ones(2, 4, 8)),
        (f"{experts}.w1.lora_B.weight", torch.ones(2, 6, 4)),
        (f"{experts}.w2.lora_A.weight", torch.ones(2, 4, 6)),
        (f"{experts}.w2.lora_B.weight", torch.ones(2, 8, 4)),
    ]
    slots = SimpleNamespace(bf16_adapter_key_style="sampler", export_bf16_adapter=lambda slot: native_export)
    members = DistributedOptimizerMembers(slots, optimizer=None, model=[], precision_aware=False)
    parametrization = AdapterParametrization(
        base_model="base",
        lora_rank=4,
        lora_alpha=8.0,
        lora_type="canonical_lora",
        target_modules=(f"{experts}.gate_proj", f"{experts}.down_proj"),
        experts_shared_outer_loras=False,
    )

    write_peft_export(tmp_path, members, 0, parametrization, targets, lora_dropout=0.0)

    with safe_open(tmp_path / "adapter_model.safetensors", framework="pt") as exported:
        assert set(exported.keys()) == {
            f"base_model.model.{experts}.{expert}.{module}.{factor}.weight"
            for expert in range(2)
            for module in ("gate_proj", "down_proj")
            for factor in ("lora_A", "lora_B")
        }
