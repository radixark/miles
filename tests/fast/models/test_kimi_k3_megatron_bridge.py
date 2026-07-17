from types import SimpleNamespace

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)

from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
from miles_plugins.megatron_bridge.kimi_k3 import KimiK3ALogMapping, KimiK3MegatronBridge
from miles_plugins.models.kimi_k3.ops import situ_and_mul
from scripts.run_kimi_k3_lora import _DEFAULT_TARGET_MODULES


def _bridge():
    bridge = object.__new__(KimiK3MegatronBridge)
    bridge.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=4,
            linear_attn_config={"num_heads": 96},
        )
    )
    return bridge


def test_situ_activation_is_registered():
    assert KimiK3MegatronBridge.hf_to_megatron_activation("situ") is situ_and_mul


def test_lora_mapping_registry_covers_safe_targets():
    registry = _bridge().mapping_registry()
    expected = {
        "decoder.layers.0.self_attention.o_proj.weight": (
            "language_model.model.layers.0.self_attn.o_proj.weight"
        ),
        "decoder.layers.3.self_attention.q_a_proj.weight": (
            "language_model.model.layers.3.self_attn.q_a_proj.weight"
        ),
        "decoder.layers.3.self_attention.kv_a_proj_with_mqa.weight": (
            "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
        ),
        "decoder.layers.1.mlp.experts.linear_fc2.weight0": (
            "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
        ),
    }

    for megatron_name, hf_name in expected.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert mapping is not None
        assert mapping.hf_param == hf_name

    expert_fc1 = registry.megatron_to_hf_lookup("decoder.layers.1.mlp.experts.linear_fc1.weight0")
    assert expert_fc1.hf_param == {
        "gate": "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
        "up": "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight",
    }


def test_mapping_registry_covers_full_kda_mla_and_moe_state():
    registry = _bridge().mapping_registry()
    kda_names = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
        "dt_bias",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "b_proj.weight",
        "g_proj.weight",
        "o_norm.weight",
        "o_proj.weight",
    ]
    mla_names = [
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "g_proj.weight",
        "o_proj.weight",
    ]
    common_layer_names = [
        "input_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "pre_mlp_layernorm.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    ]
    moe_names = [
        "mlp.router.weight",
        "mlp.router.expert_bias",
        "mlp.fc1_latent_proj.weight",
        "mlp.routed_expert_norm.weight",
        "mlp.fc2_latent_proj.weight",
        "mlp.shared_experts.linear_fc1.weight",
        "mlp.shared_experts.linear_fc2.weight",
        "mlp.experts.linear_fc1.weight0",
        "mlp.experts.linear_fc2.weight0",
    ]
    names = [
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
        *(f"decoder.layers.0.{name}" for name in common_layer_names),
        *(f"decoder.layers.0.self_attention.{name}" for name in kda_names),
        "decoder.layers.0.mlp.linear_fc1.weight",
        "decoder.layers.0.mlp.linear_fc2.weight",
        *(f"decoder.layers.1.{name}" for name in moe_names),
        *(f"decoder.layers.3.self_attention.{name}" for name in mla_names),
        "decoder.layers.3.output_attn_res_norm.weight",
        "decoder.layers.3.output_attn_res_proj.weight",
    ]

    assert all(registry.megatron_to_hf_lookup(name) is not None for name in names)


def test_custom_modules_have_explicit_parallel_mappings():
    registry = _bridge().mapping_registry()
    expected_types = {
        "decoder.layers.0.self_attention.q_proj.weight": ColumnParallelMapping,
        "decoder.layers.0.self_attention.q_conv1d.weight": ColumnParallelMapping,
        "decoder.layers.0.self_attention.A_log": KimiK3ALogMapping,
        "decoder.layers.0.self_attention.dt_bias": ColumnParallelMapping,
        "decoder.layers.0.self_attention.f_a_proj.weight": ReplicatedMapping,
        "decoder.layers.0.self_attention.o_norm.weight": ReplicatedMapping,
        "decoder.layers.0.self_attention.o_proj.weight": RowParallelMapping,
        "decoder.layers.0.self_attention_res_proj.weight": ReplicatedMapping,
        "decoder.layers.3.self_attention.q_a_proj.weight": ReplicatedMapping,
        "decoder.layers.3.self_attention.q_b_proj.weight": ColumnParallelMapping,
        "decoder.layers.1.mlp.fc1_latent_proj.weight": ReplicatedMapping,
        "decoder.layers.1.mlp.shared_experts.linear_fc1.weight": GatedMLPMapping,
        "decoder.layers.1.mlp.shared_experts.linear_fc2.weight": RowParallelMapping,
        "decoder.layers.1.mlp.experts.linear_fc2.weight0": AutoMapping,
    }

    for name, mapping_type in expected_types.items():
        assert isinstance(registry.megatron_to_hf_lookup(name), mapping_type)

    a_log = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.A_log")
    assert a_log.num_heads == 96


def test_default_targets_match_sglang_supported_modules():
    targets = _DEFAULT_TARGET_MODULES.split(",")
    assert convert_target_modules_to_hf(targets) == [
        "o_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert all("shared_experts" not in target for target in targets)
    assert all(not target.endswith(("q_proj", "k_proj", "v_proj", "g_proj")) for target in targets)
