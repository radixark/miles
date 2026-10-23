"""HF-driven K3 architecture selection, without loading TorchTitan or CUDA."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.backends.torchtitan_utils.models.kimi_k3.config import model_config_kwargs


register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


@pytest.fixture
def text_config():
    # Decoder fields from moonshotai/Kimi-K3, after BF16 checkpoint conversion.
    return SimpleNamespace(
        hidden_size=7168,
        vocab_size=163840,
        num_hidden_layers=93,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=2,
        routed_expert_hidden_size=3584,
        first_k_dense_replace=1,
        intermediate_size=33792,
        moe_intermediate_size=3072,
        attn_res_block_size=12,
        rms_norm_eps=1e-5,
        routed_scaling_factor=1.0,
        mla_use_nope=True,
        mla_use_output_gate=True,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        linear_attn_config=dict(
            full_attn_layers=[*range(4, 93, 4), 93],
            kda_layers=[i for i in range(1, 93) if i % 4],
            num_heads=96,
            head_dim=128,
            short_conv_kernel_size=4,
            gate_lower_bound=-5.0,
            use_full_rank_gate=True,
        ),
    )


def test_full_model_last_layer_is_mla_not_the_periodic_kda_default(text_config):
    config = model_config_kwargs(SimpleNamespace(text_config=text_config))
    assert (config["n_layers"], config["num_experts"]) == (93, 896)
    assert config["full_attn_layers"] == (*range(4, 93, 4), 93)
    assert (config["n_heads"], config["mla_heads"], config["head_dim"]) == (96, 96, 128)


def test_pruned_checkpoint_keeps_only_active_prefix_without_mutating_hf(text_config):
    text_config.num_hidden_layers, text_config.num_experts = 4, 64
    original = deepcopy(vars(text_config))
    config = model_config_kwargs(text_config)
    assert (config["n_layers"], config["num_experts"], config["full_attn_layers"]) == (4, 64, (4,))
    assert vars(text_config) == original


def test_mla_and_kda_dimensions_are_independent(text_config):
    text_config.num_attention_heads = 48
    text_config.q_lora_rank, text_config.kv_lora_rank = 768, 256
    text_config.rms_norm_eps, text_config.routed_scaling_factor = 1e-6, 1.5
    config = model_config_kwargs(text_config)
    assert (config["mla_heads"], config["n_heads"], config["q_lora_rank"], config["kv_lora_rank"]) == (
        48,
        96,
        768,
        256,
    )
    assert (config["norm_eps"], config["route_scale"]) == (1e-6, 1.5)


@pytest.mark.parametrize("location", ["text", "outer"])
def test_quantized_checkpoint_fails_before_allocating_a_model(text_config, location):
    outer = SimpleNamespace(text_config=text_config)
    (text_config if location == "text" else outer).quantization_config = {"format": "mxfp4-pack-quantized"}
    with pytest.raises(ValueError, match="convert_mxfp4_to_bf16"):
        model_config_kwargs(outer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mla_use_nope", False),
        ("mla_use_output_gate", False),
        ("latent_moe_use_norm", False),
        ("hidden_act", "silu"),
        ("activation_situ_beta", 3.0),
        ("moe_router_activation_func", "softmax"),
        ("moe_renormalize", False),
        ("moe_layer_freq", 2),
        ("num_expert_group", 2),
        ("num_nextn_predict_layers", 1),
        ("tie_word_embeddings", True),
    ],
)
def test_unimplemented_variants_are_rejected(text_config, field, value):
    setattr(text_config, field, value)
    with pytest.raises(ValueError, match=field):
        model_config_kwargs(text_config)


@pytest.mark.parametrize("overlap", [False, True])
def test_layer_lists_must_cover_each_active_layer_once(text_config, overlap):
    if overlap:
        text_config.linear_attn_config["kda_layers"].append(93)
    else:
        text_config.linear_attn_config["full_attn_layers"].remove(93)
    with pytest.raises(ValueError, match="partition"):
        model_config_kwargs(text_config)
