"""GLM TOP refuses the MoE options its router and experts would silently ignore."""

from types import SimpleNamespace

import pytest
import torch

glm_config = pytest.importorskip("miles_plugins.top.glm_config")


def _glm_moe_config(**overrides):
    values = dict(
        params_dtype=torch.bfloat16,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_enable_expert_bias=True,
        moe_router_fusion=False,
        moe_permute_fusion=False,
        moe_apply_probs_on_input=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        moe_shared_expert_overlap=False,
        moe_shared_expert_gate=False,
        num_moe_experts=256,
        expert_model_parallel_size=4,
        hidden_size=6144,
        moe_ffn_hidden_size=2048,
        moe_router_topk_scaling_factor=2.5,
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        moe_router_force_load_balancing=False,
        moe_router_dtype=None,
        moe_grouped_gemm=False,
    )
    return SimpleNamespace(**(values | overrides))


def test_the_recipe_configuration_is_admitted():
    glm_config.validate_glm_moe_config(_glm_moe_config())


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"moe_aux_loss_coeff": 1e-3}, "aux loss"),
        ({"moe_aux_loss_coeff": [0.0, 1e-3]}, "aux loss"),
        ({"moe_z_loss_coeff": 1e-3}, "z-loss"),
        ({"moe_router_force_load_balancing": True}, "force-load-balancing"),
        ({"moe_router_dtype": "fp32"}, "router-dtype"),
        ({"moe_grouped_gemm": True}, "moe-grouped-gemm"),
    ],
    ids=["aux-loss", "aux-loss-list", "z-loss", "force-load-balancing", "router-dtype", "grouped-gemm"],
)
def test_options_the_top_moe_does_not_apply_are_refused(overrides, refusal):
    with pytest.raises(NotImplementedError, match=refusal):
        glm_config.validate_glm_moe_config(_glm_moe_config(**overrides))


def _glm_pair(**hf_overrides):
    """A 5-layer GLM-5.2 slice: the trainer config and its HF config, as the TOP args build them."""
    hf = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        hidden_size=6144,
        n_routed_experts=256,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        num_hidden_layers=5,
        num_attention_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        v_head_dim=256,
        qk_rope_head_dim=64,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        index_head_dim=128,
        rope_theta=1e6,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=["full", "full", "full", "shared", "shared"],
        index_kpool=None,
    )
    config = SimpleNamespace(
        hidden_size=6144,
        num_moe_experts=256,
        moe_ffn_hidden_size=2048,
        moe_router_topk=8,
        num_layers=5,
        num_attention_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_head_dim=192,
        v_head_dim=256,
        qk_pos_emb_head_dim=64,
        moe_shared_expert_intermediate_size=2048,
        moe_router_topk_scaling_factor=2.5,
        rotary_scaling_factor=1.0,
        rotary_base=1e6,
    )
    return config, SimpleNamespace(**(hf | hf_overrides))


@pytest.mark.parametrize(
    ("hf_overrides", "refusal"),
    [
        ({}, None),
        ({"indexer_types": ["full"] * 5}, "rule the trainer runs"),
        ({"indexer_types": None, "index_topk_pattern": "FFFFS"}, "rule the trainer runs"),
        ({"index_kpool": 4}, "index_kpool"),
    ],
    ids=["glm5-checkpoint", "indexer-types", "topk-pattern", "kpool"],
)
def test_index_sharing_must_match_the_trainer_rule(hf_overrides, refusal):
    """Serving selects keys by these HF fields; the trainer only by index_topk_freq/offset."""
    from unittest.mock import patch

    config, hf = _glm_pair(**hf_overrides)
    args = SimpleNamespace(hf_checkpoint="/ckpt", miles_dsa_topk_backend="flashinfer")
    with (
        patch("miles.utils.hf_utils.config.load_hf_config", return_value=hf),
        patch("sglang.srt.true_on_policy.glm.validate_envs"),
    ):
        if refusal is None:
            glm_config.validate_glm_config(args, config)
        else:
            with pytest.raises(NotImplementedError, match=refusal):
                glm_config.validate_glm_config(args, config)
