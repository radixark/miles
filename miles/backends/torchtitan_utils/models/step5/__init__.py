from dataclasses import dataclass
from functools import partial

from torch import nn
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.config_utils import make_moe_config, make_routed_experts_config, make_router_config
from torchtitan.protocols.model_spec import ModelSpec

from miles.backends.torchtitan_utils.models.step5.layers import (
    Step5FeedForward,
    Step5FinalRMSNorm,
    Step5GroupedExperts,
    Step5Indexer,
    Step5RMSNorm,
    Step5SlidingWindowAttention,
    Step5SparseAttention,
)
from miles.backends.torchtitan_utils.models.step5.model import Step5Block, Step5Model
from miles.backends.torchtitan_utils.models.step5.parallelize import parallelize_step5
from miles.backends.torchtitan_utils.models.step5.state_dict_adapter import Step5StateDictAdapter
from miles_plugins.models.step5.indexer_ops import INDEXER_HEAD_DIM, INDEXER_ROPE_DIM
from miles_plugins.models.step5.layer_math import FULL_ATTENTION_ROPE, SLIDING_WINDOW_ROPE, SWIGLU_LIMIT
from miles_plugins.models.step5.region_selection import REGION_SIZE, TOP_REGIONS
from miles_plugins.models.step5.region_selector import INDEXER_NUM_HEADS, INDEXER_ROPE

__all__ = ["build_model_config", "model_registry"]

_EXPERTS_INIT = {name: partial(nn.init.trunc_normal_, std=0.02) for name in ("w1_EFD", "w2_EDF", "w3_EFD")}
_ROPE = {"sliding_attention": SLIDING_WINDOW_ROPE, "full_attention": FULL_ATTENTION_ROPE}


@dataclass(frozen=True)
class _Architecture:
    """The Step5 text decoder's shape, read from the checkpoint's config.json."""

    dim: int
    vocab_size: int
    norm_eps: float
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    moe_layers: frozenset[int]
    dense_hidden_dim: int
    moe_hidden_dim: int
    shared_hidden_dim: int
    num_experts: int
    top_k: int
    route_scale: float
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, hf_config) -> "_Architecture":
        text = hf_config.text_config
        _require_supported(text)
        return cls(
            dim=text.hidden_size,
            vocab_size=text.vocab_size,
            norm_eps=text.rms_norm_eps,
            max_position_embeddings=text.max_position_embeddings,
            sliding_window=text.sliding_window,
            layer_types=tuple(text.layer_types),
            moe_layers=frozenset(text.moe_layer_list),
            dense_hidden_dim=text.intermediate_size,
            moe_hidden_dim=text.moe_intermediate_size,
            shared_hidden_dim=text.share_expert_dim,
            num_experts=text.moe_num_experts,
            top_k=text.moe_top_k,
            route_scale=text.moe_router_scaling_factor,
            n_heads=text.num_attention_heads,
            n_kv_heads=text.num_attention_groups,
            head_dim=text.head_dim,
        )


def _require_supported(text) -> None:
    """The variant this package implements, as the Megatron model checks it; anything else would build
    the wrong model silently."""
    sparse = text.sparse_config
    scaling = text.rope_scaling or {}
    unsupported = {
        "zero_centered norms": text.zero_centered is not True,
        "an fp32 residual connection": text.fp32_residual_connection is not True,
        "the head-wise attention gate": text.use_head_wise_attn_gate is not True,
        "sliding-window and full-attention layers only": not set(text.layer_types) <= set(_ROPE),
        "equal sliding-window and full-attention head counts": text.swa_num_attention_heads
        not in (None, text.num_attention_heads),
        "each layer's RoPE theta and rotary fraction of its attention type": [
            (theta, fraction) for theta, fraction in zip(text.rope_theta, text.partial_rotary_factors, strict=True)
        ]
        != [(_ROPE[kind].theta, _ROPE[kind].rotary_percent) for kind in text.layer_types],
        "llama3 rope_scaling with factor 1": scaling.get("factor", 1.0) != 1.0,
        "a selection-only router bias": text.use_moe_router_bias is not True,
        "renormalized expert weights": text.norm_expert_weight is not True,
        f"every SwiGLU clamped to {SWIGLU_LIMIT}": set(text.swiglu_limits) | set(text.swiglu_limits_shared)
        != {SWIGLU_LIMIT},
        "the frozen region indexer of the Megatron model": {
            "enabled": sparse.get("enabled"),
            "apply_to_layer_types": sparse.get("apply_to_layer_types"),
            "num_provider_groups": sparse.get("num_provider_groups"),
            "sparse_indexer_num_heads": sparse.get("sparse_indexer_num_heads"),
            "proxy_dim": sparse.get("proxy_dim"),
            "region_block_size": sparse.get("region_block_size"),
            "topk": sparse.get("topk"),
            "sparse_indexer_rope_dim": sparse.get("sparse_indexer_rope_dim"),
            "rope_scaling": (
                scaling.get("low_freq_factor"),
                scaling.get("high_freq_factor"),
                scaling.get("original_max_position_embeddings"),
            ),
        }
        != {
            "enabled": True,
            "apply_to_layer_types": ["full_attention"],
            "num_provider_groups": text.num_attention_groups,
            "sparse_indexer_num_heads": INDEXER_NUM_HEADS,
            "proxy_dim": INDEXER_HEAD_DIM,
            "region_block_size": REGION_SIZE,
            "topk": TOP_REGIONS,
            "sparse_indexer_rope_dim": INDEXER_ROPE_DIM,
            "rope_scaling": (
                INDEXER_ROPE.low_freq_factor,
                INDEXER_ROPE.high_freq_factor,
                INDEXER_ROPE.original_max_position_embeddings,
            ),
        },
    }
    failed = [requirement for requirement, is_unsupported in unsupported.items() if is_unsupported]
    if failed:
        raise ValueError(f"step5 needs {'; '.join(failed)}")


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _ffn(arch: _Architecture, hidden_dim: int) -> Step5FeedForward.Config:
    return Step5FeedForward.Config(
        w1=_linear(arch.dim, hidden_dim),
        w2=_linear(hidden_dim, arch.dim),
        w3=_linear(arch.dim, hidden_dim),
        swiglu_limit=SWIGLU_LIMIT,
    )


def _moe(arch: _Architecture):
    routed = make_routed_experts_config(
        dim=arch.dim,
        hidden_dim=arch.moe_hidden_dim,
        num_experts=arch.num_experts,
        top_k=arch.top_k,
        param_init=_EXPERTS_INIT,
        comm_backend="standard",
    )
    routed.inner_experts = Step5GroupedExperts.Config(
        dim=arch.dim,
        hidden_dim=arch.moe_hidden_dim,
        num_experts=arch.num_experts,
        param_init=_EXPERTS_INIT,
        swiglu_limit=SWIGLU_LIMIT,
    )
    return make_moe_config(
        num_experts=arch.num_experts,
        router=make_router_config(
            dim=arch.dim,
            num_experts=arch.num_experts,
            gate_param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
            top_k=arch.top_k,
            score_func="sigmoid",
            route_norm=True,
            route_scale=arch.route_scale,
        ),
        routed_experts=routed,
        shared_experts=_ffn(arch, arch.shared_hidden_dim),
    )


def _attention(arch: _Architecture, *, layer_type: str):
    q_dim, kv_dim = arch.n_heads * arch.head_dim, arch.n_kv_heads * arch.head_dim
    common = dict(
        n_heads=arch.n_heads,
        n_kv_heads=arch.n_kv_heads,
        head_dim=arch.head_dim,
        rope=_ROPE[layer_type],
        q_proj=_linear(arch.dim, q_dim),
        k_proj=_linear(arch.dim, kv_dim),
        v_proj=_linear(arch.dim, kv_dim),
        g_proj=_linear(arch.dim, arch.n_heads),
        o_proj=_linear(q_dim, arch.dim),
        q_norm=Step5RMSNorm.Config(dim=arch.head_dim, eps=arch.norm_eps),
        k_norm=Step5RMSNorm.Config(dim=arch.head_dim, eps=arch.norm_eps),
    )
    if layer_type == "sliding_attention":
        return Step5SlidingWindowAttention.Config(**common, window=arch.sliding_window)
    indexer = Step5Indexer.Config(
        dim=arch.dim, num_heads=INDEXER_NUM_HEADS, num_attention_heads=arch.n_heads, eps=arch.norm_eps
    )
    return Step5SparseAttention.Config(**common, indexer=indexer)


def _block(arch: _Architecture, index: int) -> Step5Block.Config:
    moe = index in arch.moe_layers
    return Step5Block.Config(
        attention_norm=Step5RMSNorm.Config(dim=arch.dim, eps=arch.norm_eps),
        ffn_norm=Step5RMSNorm.Config(dim=arch.dim, eps=arch.norm_eps),
        attention=_attention(arch, layer_type=arch.layer_types[index]),
        feed_forward=None if moe else _ffn(arch, arch.dense_hidden_dim),
        moe=_moe(arch) if moe else None,
    )


def build_model_config(hf_config) -> Step5Model.Config:
    arch = _Architecture.from_hf_config(hf_config)
    return Step5Model.Config(
        dim=arch.dim,
        vocab_size=arch.vocab_size,
        max_position_embeddings=arch.max_position_embeddings,
        sliding_window=arch.sliding_window,
        tok_embeddings=Embedding.Config(num_embeddings=arch.vocab_size, embedding_dim=arch.dim),
        norm=Step5FinalRMSNorm.Config(rms=Step5RMSNorm.Config(dim=arch.dim, eps=arch.norm_eps)),
        lm_head=_linear(arch.dim, arch.vocab_size),
        layers=[_block(arch, index) for index in range(len(arch.layer_types))],
    )


def model_registry(flavor: str, attn_backend: str = "flex", *, hf_config) -> ModelSpec:
    """Sized from the checkpoint's config.json, as Megatron is; ``flavor`` only names the run."""
    return ModelSpec(
        name="step5",
        flavor=flavor,
        model=build_model_config(hf_config),
        parallelize_fn=parallelize_step5,
        pipelining_fn=pipeline_llm,
        # the router bias only steers selection and stays as released, as Megatron's update rate of 0 keeps it
        post_optimizer_build_fn=None,
        state_dict_adapter=Step5StateDictAdapter,
    )
