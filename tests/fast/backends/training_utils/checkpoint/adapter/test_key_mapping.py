"""PEFT and official Tinker names map to HF keys and back without loss."""

import pytest

from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    HfKey,
    is_tinker_only_name,
    map_hf_keys_to_peft_keys,
    map_hf_keys_to_tinker_keys,
    map_peft_keys_to_hf_keys,
    map_tinker_keys_to_hf_keys,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets


def test_per_expert_peft_names_stack_under_one_hf_key():
    name = "model.layers.3.mlp.experts.17.gate_proj.lora_A.weight"
    assert map_peft_keys_to_hf_keys(name) == HfKey("model.layers.3.mlp.experts.gate_proj.lora_A", 17)
    assert map_peft_keys_to_hf_keys(f"base_model.model.{name}") == map_peft_keys_to_hf_keys(name)
    assert map_hf_keys_to_peft_keys("model.layers.3.mlp.experts.gate_proj.lora_A", 17) == name, "main's bare names"


@pytest.mark.parametrize(
    "model_type, tinker_name, hf_key",
    [
        ("qwen3", "base_model.model.model.unembed_tokens.lora_B.weight", "lm_head.lora_B"),
        (
            "qwen3",
            "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight",
            "model.layers.0.mlp.experts.up_proj.lora_A",
        ),
        (
            "gpt_oss",
            "base_model.model.model.layers.2.attn.q_proj.lora_A.weight",
            "model.layers.2.self_attn.q_proj.lora_A",
        ),
    ],
)
def test_tinker_names_round_trip(model_type, tinker_name, hf_key):
    targets = get_hf_lora_targets({"model_type": model_type, "num_hidden_layers": 4})
    assert map_tinker_keys_to_hf_keys(tinker_name, targets) == HfKey(hf_key)
    assert map_hf_keys_to_tinker_keys(hf_key, targets) == tinker_name
    assert is_tinker_only_name(tinker_name, targets) == (
        hf_key.startswith("lm_head") or ".experts." in hf_key or model_type == "gpt_oss"
    )


def test_tinker_names_gain_the_vl_layer_prefix():
    text_config = {
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 2,
    }
    targets = get_hf_lora_targets({"model_type": "qwen3_5", "text_config": text_config})
    name = "base_model.model.model.layers.1.linear_attn.in_proj_q.lora_A.weight"
    assert (
        map_tinker_keys_to_hf_keys(name, targets).key == "model.language_model.layers.1.linear_attn.in_proj_q.lora_A"
    )
    assert is_tinker_only_name(name, targets), "a dense export without a head is still Tinker-named"
    assert not is_tinker_only_name("model.language_model.layers.1.linear_attn.in_proj_q.lora_A.weight", targets)
