"""Validate model dimensions, per-layer structure, and the supported indexer profile against the checkpoint."""

from typing import Any

from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.step5.indexer import INDEXER_NUM_HEADS, INDEXER_ROPE
from miles_plugins.models.step5.indexer_ops import INDEXER_HEAD_DIM, INDEXER_ROPE_DIM
from miles_plugins.models.step5.layers import (
    FULL_ATTENTION_ROPE,
    SLIDING_WINDOW_ROPE,
    SWIGLU_LIMIT,
    is_full_attention_layer,
)
from miles_plugins.models.step5.region_selection import REGION_SIZE, TOP_REGIONS


def check_matches_checkpoint(config: TransformerConfig, text_config: Any) -> None:
    """Raise unless ``config`` describes every layer of ``text_config`` (SGLang's ``Step5Config``)."""
    mismatches = [
        f"{name}: checkpoint {actual!r}, megatron {expected!r}"
        for name, actual, expected in (
            _global_fields(config, text_config)
            + _indexer_fields(config, text_config)
            + _layer_fields(config, text_config)
        )
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Step5 model flags do not match the checkpoint config:\n  " + "\n  ".join(mismatches))


def _global_fields(config: TransformerConfig, text_config: Any) -> list[tuple[str, Any, Any]]:
    assert config.window_size is not None, "Step5 needs --window-size 511,0 for its sliding-window layers"
    window_left, window_right = config.window_size
    return [
        ("num_hidden_layers", text_config.num_hidden_layers, config.num_layers),
        ("hidden_size", text_config.hidden_size, config.hidden_size),
        ("intermediate_size", text_config.intermediate_size, config.ffn_hidden_size),
        ("num_attention_heads", text_config.num_attention_heads, config.num_attention_heads),
        (
            "swa_num_attention_heads",
            text_config.swa_num_attention_heads or text_config.num_attention_heads,
            config.num_attention_heads,
        ),
        ("num_attention_groups", text_config.num_attention_groups, config.num_query_groups),
        ("head_dim", text_config.head_dim, config.kv_channels),
        ("rms_norm_eps", text_config.rms_norm_eps, config.layernorm_epsilon),
        ("zero_centered", text_config.zero_centered, True),
        ("fp32_residual_connection", text_config.fp32_residual_connection, config.fp32_residual_connection),
        ("use_head_wise_attn_gate", text_config.use_head_wise_attn_gate, config.head_wise_attn_gate),
        # A window of W keys is the query plus W - 1 before it.
        ("sliding_window", text_config.sliding_window, window_left + 1 if window_right == 0 else None),
        # llama3 scaling with factor 1 leaves the full-attention frequencies unchanged.
        ("rope_scaling.factor", (text_config.rope_scaling or {}).get("factor", 1.0), 1.0),
        ("moe_num_experts", text_config.moe_num_experts, config.num_moe_experts),
        ("moe_top_k", text_config.moe_top_k, config.moe_router_topk),
        ("moe_intermediate_size", text_config.moe_intermediate_size, config.moe_ffn_hidden_size),
        ("share_expert_dim", text_config.share_expert_dim, config.moe_shared_expert_intermediate_size),
        ("moe_router_scaling_factor", text_config.moe_router_scaling_factor, config.moe_router_topk_scaling_factor),
        ("use_moe_router_bias", text_config.use_moe_router_bias, config.moe_router_enable_expert_bias),
        ("norm_expert_weight", text_config.norm_expert_weight, True),
    ]


def _indexer_fields(config: TransformerConfig, text_config: Any) -> list[tuple[str, Any, Any]]:
    sparse = text_config.sparse_config
    scaling = text_config.rope_scaling or {}
    expected_sparse = {
        "enabled": True,
        "apply_to_layer_types": ["full_attention"],
        "num_provider_groups": config.num_query_groups,
        "sparse_indexer_num_heads": INDEXER_NUM_HEADS,
        "sparse_indexer_num_k_heads": 1,
        "proxy_dim": INDEXER_HEAD_DIM,
        "region_block_size": REGION_SIZE,
        "topk": TOP_REGIONS,
        "compression_method": "csa_block_compress",
        "sparse_indexer_use_rope": True,
        "sparse_indexer_rope_dim": INDEXER_ROPE_DIM,
        "sparse_indexer_q_norm_type": "rmsnorm",
        "sparse_indexer_k_norm_type": "layernorm",
        "sparse_indexer_csa_z_norm_type": "none",
    }
    expected_scaling = {
        "rope_type": "llama3",
        "low_freq_factor": INDEXER_ROPE.low_freq_factor,
        "high_freq_factor": INDEXER_ROPE.high_freq_factor,
        "original_max_position_embeddings": INDEXER_ROPE.original_max_position_embeddings,
    }
    return [(f"sparse_config.{key}", sparse.get(key), value) for key, value in expected_sparse.items()] + [
        (f"rope_scaling.{key}", scaling.get(key), value) for key, value in expected_scaling.items()
    ]


def _layer_fields(config: TransformerConfig, text_config: Any) -> list[tuple[str, Any, Any]]:
    assert isinstance(config.moe_layer_freq, list), "Step5 needs an explicit per-layer --moe-layer-freq"
    moe_layers = set(text_config.moe_layer_list)
    fields = []
    for index in range(config.num_layers):
        full = is_full_attention_layer(config, index + 1)
        rope = FULL_ATTENTION_ROPE if full else SLIDING_WINDOW_ROPE
        fields += [
            (
                f"layer_types[{index}]",
                text_config.layer_types[index],
                "full_attention" if full else "sliding_attention",
            ),
            (f"rope_theta[{index}]", text_config.rope_theta[index], rope.theta),
            (f"partial_rotary_factors[{index}]", text_config.partial_rotary_factors[index], rope.rotary_percent),
            (f"moe[{index}]", index in moe_layers, bool(config.moe_layer_freq[index])),
            (f"swiglu_limits[{index}]", text_config.swiglu_limits[index], SWIGLU_LIMIT),
            (f"swiglu_limits_shared[{index}]", text_config.swiglu_limits_shared[index], SWIGLU_LIMIT),
        ]
    return fields
