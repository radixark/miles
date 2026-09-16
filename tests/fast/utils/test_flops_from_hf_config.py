from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.utils.flops_utils import calculate_fwd_flops, flops_args_from_hf_config

SEQLENS = [1024, 3072]


def megatron_args(**overrides):
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=4,
        vocab_size=152064,
        num_layers=8,
        ffn_hidden_size=8192,
        kv_channels=128,
        num_experts=None,
        moe_ffn_hidden_size=None,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=0,
        qk_pos_emb_head_dim=0,
        v_head_dim=0,
    )
    return SimpleNamespace(**(base | overrides))


def hf_config(**overrides):
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=152064,
        num_hidden_layers=8,
        intermediate_size=8192,
        head_dim=128,
    )
    return SimpleNamespace(**(base | overrides))


def assert_same(hf, megatron):
    adapted = flops_args_from_hf_config(hf)
    assert calculate_fwd_flops(SEQLENS, adapted) == calculate_fwd_flops(SEQLENS, megatron)


def test_dense_gqa_model_matches():
    assert_same(hf_config(), megatron_args())


def test_head_dim_falls_back_to_hidden_over_heads():
    hf = hf_config()
    del hf.head_dim
    assert_same(hf, megatron_args(kv_channels=2048 // 16))


def test_missing_kv_heads_means_full_multi_head():
    hf = hf_config()
    del hf.num_key_value_heads
    assert_same(hf, megatron_args(num_query_groups=16))


def test_every_layer_moe_matches():
    assert_same(
        hf_config(num_experts=64, moe_intermediate_size=1024, num_experts_per_tok=8),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_mixtral_style_experts_sized_by_plain_intermediate_size():
    assert_same(
        hf_config(num_local_experts=8, num_experts_per_tok=2),
        megatron_args(
            num_experts=8,
            moe_ffn_hidden_size=8192,
            moe_router_topk=2,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_gpt_oss_style_experts_sized_by_plain_intermediate_size():
    assert_same(
        hf_config(num_local_experts=32, num_experts_per_tok=4, intermediate_size=2880),
        megatron_args(
            num_experts=32,
            ffn_hidden_size=2880,
            moe_ffn_hidden_size=2880,
            moe_router_topk=4,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_expert_width_is_not_borrowed_when_the_config_declares_one():
    adapted = flops_args_from_hf_config(
        hf_config(num_local_experts=8, num_experts_per_tok=2, moe_intermediate_size=1024)
    )
    assert adapted.moe_ffn_hidden_size == 1024


def test_dense_config_keeps_no_expert_width():
    assert flops_args_from_hf_config(hf_config()).moe_ffn_hidden_size is None


def test_all_moe_config_without_a_dense_ffn_width():
    hf = hf_config(num_experts=256, moe_intermediate_size=512, num_experts_per_tok=8)
    del hf.intermediate_size
    assert_same(
        hf,
        megatron_args(
            ffn_hidden_size=None,
            num_experts=256,
            moe_ffn_hidden_size=512,
            moe_router_topk=8,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_shared_experts_sized_from_the_borrowed_expert_width():
    hf = hf_config(n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=2, intermediate_size=2048)
    assert_same(
        hf,
        megatron_args(
            ffn_hidden_size=2048,
            num_experts=256,
            moe_ffn_hidden_size=2048,
            moe_router_topk=6,
            moe_shared_expert_intermediate_size=2 * 2048,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_a_config_that_hides_its_ffn_width_is_rejected_up_front():
    hf = hf_config()
    del hf.intermediate_size
    with pytest.raises(ValueError, match="does not expose the FFN widths"):
        flops_args_from_hf_config(hf)


def test_deepseek_style_dense_prefix_matches():
    assert_same(
        hf_config(
            n_routed_experts=64,
            moe_intermediate_size=1024,
            num_experts_per_tok=8,
            first_k_dense_replace=3,
            n_shared_experts=2,
        ),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_shared_expert_intermediate_size=2 * 1024,
            moe_layer_freq=[0, 0, 0, 1, 1, 1, 1, 1],
        ),
    )


def test_qwen_style_sparse_step_and_dense_layers_match():
    assert_same(
        hf_config(
            num_experts=64,
            moe_intermediate_size=1024,
            num_experts_per_tok=8,
            decoder_sparse_step=2,
            mlp_only_layers=[5],
            shared_expert_intermediate_size=2048,
        ),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_shared_expert_intermediate_size=2048,
            moe_layer_freq=[0, 1, 0, 1, 0, 0, 0, 1],
        ),
    )


def test_mla_matches():
    assert_same(
        hf_config(
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        megatron_args(
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
        ),
    )


def test_moe_layers_actually_cost_more_than_dense():
    dense = calculate_fwd_flops(SEQLENS, flops_args_from_hf_config(hf_config()))
    moe = calculate_fwd_flops(
        SEQLENS,
        flops_args_from_hf_config(hf_config(num_experts=64, moe_intermediate_size=8192, num_experts_per_tok=4)),
    )
    assert moe > dense


@pytest.mark.parametrize("first_dense", [0, 8, 99])
def test_dense_prefix_clamps_to_the_layer_count(first_dense):
    args = flops_args_from_hf_config(
        hf_config(
            n_routed_experts=64, moe_intermediate_size=1024, num_experts_per_tok=8, first_k_dense_replace=first_dense
        )
    )
    assert len(args.moe_layer_freq) == 8


def test_multimodal_config_sizes_the_text_tower_not_the_vision_tower():
    vlm = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        text_config=SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=152064,
            num_hidden_layers=8,
            intermediate_size=8192,
            head_dim=128,
        ),
    )
    assert_same(vlm, megatron_args())


def test_get_text_config_wins_over_a_text_config_attribute():
    text = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=152064,
        num_hidden_layers=8,
        intermediate_size=8192,
        head_dim=128,
    )
    cfg = SimpleNamespace(num_hidden_layers=1, hidden_size=64, num_attention_heads=4, get_text_config=lambda: text)
    assert flops_args_from_hf_config(cfg).num_layers == 8
