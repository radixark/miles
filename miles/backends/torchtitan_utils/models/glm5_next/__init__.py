from dataclasses import dataclass
from functools import partial

from torch import nn
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.config_utils import make_moe_config, make_routed_experts_config, make_router_config
from torchtitan.protocols.model_spec import ModelSpec

from miles.backends.torchtitan_utils.models.glm5_next.layers import (
    ClampedFeedForward,
    ClampedGroupedExperts,
    DSAAttention,
    GatedRMSNorm,
    Glm5NextLayerNorm,
    Glm5NextRMSNorm,
    HyperConnection,
    KDAGate,
    KimiDeltaAttention,
    KpoolIndexer,
    ShortConv,
)
from miles.backends.torchtitan_utils.models.glm5_next.model import Glm5NextBlock, Glm5NextModel
from miles.backends.torchtitan_utils.models.glm5_next.parallelize import parallelize_glm5_next
from miles.backends.torchtitan_utils.models.glm5_next.state_dict_adapter import Glm5NextStateDictAdapter

__all__ = ["model_registry"]

_EXPERTS_INIT = {name: partial(nn.init.trunc_normal_, std=0.02) for name in ("w1_EFD", "w2_EDF", "w3_EFD")}
# Megatron hardcodes the indexer's key LayerNorm eps; the HF config does not carry it
_INDEXER_K_NORM_EPS = 1e-6


@dataclass(frozen=True)
class _Architecture:
    """The GLM-5.3-Flash text decoder's shape, read from the checkpoint's config.json."""

    dim: int
    vocab_size: int
    norm_eps: float
    max_position_embeddings: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    dense_hidden_dim: int
    moe_hidden_dim: int
    num_experts: int
    num_shared_experts: int
    top_k: int
    route_scale: float
    route_norm: bool
    swiglu_limit: float
    num_streams: int
    sinkhorn_iterations: int
    hc_eps: float
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    index_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    kda_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    kda_gate_lower_bound: float

    @classmethod
    def from_hf_config(cls, hf_config) -> "_Architecture":
        text = getattr(hf_config, "text_config", None) or hf_config
        _require_supported(text)
        linear = text.linear_attn_config
        return cls(
            dim=text.hidden_size,
            vocab_size=text.vocab_size,
            norm_eps=text.rms_norm_eps,
            max_position_embeddings=text.max_position_embeddings,
            layer_types=tuple(text.layer_types),
            mlp_layer_types=tuple(text.mlp_layer_types),
            dense_hidden_dim=text.intermediate_size,
            moe_hidden_dim=text.moe_intermediate_size,
            num_experts=text.n_routed_experts,
            num_shared_experts=text.n_shared_experts,
            top_k=text.num_experts_per_tok,
            route_scale=text.routed_scaling_factor,
            route_norm=text.norm_topk_prob,
            swiglu_limit=text.swiglu_limit,
            num_streams=text.hc_mult,
            sinkhorn_iterations=text.hc_sinkhorn_iters,
            hc_eps=text.hc_eps,
            n_heads=text.num_attention_heads,
            q_lora_rank=text.q_lora_rank,
            kv_lora_rank=text.kv_lora_rank,
            qk_head_dim=text.qk_head_dim,
            v_head_dim=text.v_head_dim,
            index_heads=text.index_n_heads,
            index_head_dim=text.index_head_dim,
            index_topk=text.index_topk,
            index_kpool=text.index_kpool,
            kda_heads=linear["num_heads"],
            kda_head_dim=linear["head_dim"],
            kda_conv_kernel_size=linear["short_conv_kernel_size"],
            kda_gate_lower_bound=linear["gate_lower_bound"],
        )


def _require_supported(text) -> None:
    """The variants this package implements; anything else would build the wrong model silently."""
    unsupported = {
        "qk_rope_head_dim must be 0 (NoPE MLA)": text.qk_rope_head_dim != 0,
        "scoring_func must be sigmoid": text.scoring_func != "sigmoid",
        "n_group / topk_group must be 1 (no group-limited routing)": (text.n_group, text.topk_group) != (1, 1),
        "the kpool indexer must compress and always select the tail": not (
            text.index_kpool > 1 and text.index_kpool_compress and text.index_kpool_always_select_tail
        ),
        "mhc must be on": not text.mhc,
        "layer_types must be linear_attention / deepseek_sparse_attention": not set(text.layer_types)
        <= {"linear_attention", "deepseek_sparse_attention"},
        "mlp_layer_types must be dense / sparse": not set(text.mlp_layer_types) <= {"dense", "sparse"},
    }
    failed = [reason for reason, is_unsupported in unsupported.items() if is_unsupported]
    if failed:
        raise ValueError(f"glm5_next cannot build this checkpoint: {'; '.join(failed)}")


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _ffn(arch: _Architecture, hidden_dim: int) -> ClampedFeedForward.Config:
    return ClampedFeedForward.Config(
        w1=_linear(arch.dim, hidden_dim),
        w2=_linear(hidden_dim, arch.dim),
        w3=_linear(arch.dim, hidden_dim),
        swiglu_limit=arch.swiglu_limit,
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
    routed.inner_experts = ClampedGroupedExperts.Config(
        dim=arch.dim,
        hidden_dim=arch.moe_hidden_dim,
        num_experts=arch.num_experts,
        param_init=_EXPERTS_INIT,
        swiglu_limit=arch.swiglu_limit,
    )
    return make_moe_config(
        num_experts=arch.num_experts,
        router=make_router_config(
            dim=arch.dim,
            num_experts=arch.num_experts,
            gate_param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
            top_k=arch.top_k,
            score_func="sigmoid",
            route_norm=arch.route_norm,
            route_scale=arch.route_scale,
        ),
        routed_experts=routed,
        shared_experts=_ffn(arch, arch.moe_hidden_dim * arch.num_shared_experts),
    )


def _kda(arch: _Architecture) -> KimiDeltaAttention.Config:
    num_heads, head_dim = arch.kda_heads, arch.kda_head_dim
    proj = num_heads * head_dim
    conv = ShortConv.Config(channels=proj, kernel_size=arch.kda_conv_kernel_size)
    return KimiDeltaAttention.Config(
        num_heads=num_heads,
        head_dim=head_dim,
        q_proj=_linear(arch.dim, proj),
        k_proj=_linear(arch.dim, proj),
        v_proj=_linear(arch.dim, proj),
        q_conv1d=conv,
        k_conv1d=conv,
        v_conv1d=conv,
        b_proj=_linear(arch.dim, num_heads),
        f_a_proj=_linear(arch.dim, head_dim),
        f_b_proj=_linear(head_dim, proj),
        g_a_proj=_linear(arch.dim, head_dim),
        g_b_proj=_linear(head_dim, proj),
        gate=KDAGate.Config(num_heads=num_heads, head_dim=head_dim, lower_bound=arch.kda_gate_lower_bound),
        o_norm=GatedRMSNorm.Config(dim=head_dim, eps=arch.norm_eps),
        o_proj=_linear(proj, arch.dim),
    )


def _dsa(arch: _Architecture) -> DSAAttention.Config:
    return DSAAttention.Config(
        n_heads=arch.n_heads,
        kv_lora_rank=arch.kv_lora_rank,
        qk_head_dim=arch.qk_head_dim,
        v_head_dim=arch.v_head_dim,
        wq_a=_linear(arch.dim, arch.q_lora_rank),
        q_norm=Glm5NextRMSNorm.Config(dim=arch.q_lora_rank, eps=arch.norm_eps),
        wq_b=_linear(arch.q_lora_rank, arch.n_heads * arch.qk_head_dim),
        wkv_a=_linear(arch.dim, arch.kv_lora_rank),
        kv_norm=Glm5NextRMSNorm.Config(dim=arch.kv_lora_rank, eps=arch.norm_eps),
        wkv_b=_linear(arch.kv_lora_rank, arch.n_heads * (arch.qk_head_dim + arch.v_head_dim)),
        o_proj=_linear(arch.n_heads * arch.v_head_dim, arch.dim),
        indexer=KpoolIndexer.Config(
            dim=arch.dim,
            num_heads=arch.index_heads,
            head_dim=arch.index_head_dim,
            topk=arch.index_topk,
            kpool=arch.index_kpool,
            wq_b=_linear(arch.q_lora_rank, arch.index_heads * arch.index_head_dim),
            wk=_linear(arch.dim, arch.index_head_dim),
            k_norm=Glm5NextLayerNorm.Config(dim=arch.index_head_dim, eps=_INDEXER_K_NORM_EPS),
            weights_proj=_linear(arch.dim, arch.index_heads),
        ),
    )


def _hyper_connection(arch: _Architecture) -> HyperConnection.Config:
    return HyperConnection.Config(
        dim=arch.dim,
        num_streams=arch.num_streams,
        sinkhorn_iterations=arch.sinkhorn_iterations,
        eps=arch.hc_eps,
        norm_eps=arch.norm_eps,
    )


def _block(arch: _Architecture, *, layer_type: str, mlp_type: str) -> Glm5NextBlock.Config:
    linear_attention = layer_type == "linear_attention"
    dense_mlp = mlp_type == "dense"
    return Glm5NextBlock.Config(
        hc_attn=_hyper_connection(arch),
        hc_ffn=_hyper_connection(arch),
        attention_norm=Glm5NextRMSNorm.Config(dim=arch.dim, eps=arch.norm_eps),
        ffn_norm=Glm5NextRMSNorm.Config(dim=arch.dim, eps=arch.norm_eps),
        kda=_kda(arch) if linear_attention else None,
        attention=None if linear_attention else _dsa(arch),
        feed_forward=_ffn(arch, arch.dense_hidden_dim) if dense_mlp else None,
        moe=None if dense_mlp else _moe(arch),
    )


def build_model_config(hf_config) -> Glm5NextModel.Config:
    arch = _Architecture.from_hf_config(hf_config)
    return Glm5NextModel.Config(
        dim=arch.dim,
        vocab_size=arch.vocab_size,
        num_streams=arch.num_streams,
        max_position_embeddings=arch.max_position_embeddings,
        tok_embeddings=Embedding.Config(num_embeddings=arch.vocab_size, embedding_dim=arch.dim),
        norm=Glm5NextRMSNorm.Config(dim=arch.dim, eps=arch.norm_eps),
        lm_head=_linear(arch.dim, arch.vocab_size),
        layers=[
            _block(arch, layer_type=layer_type, mlp_type=mlp_type)
            for layer_type, mlp_type in zip(arch.layer_types, arch.mlp_layer_types, strict=True)
        ],
    )


def model_registry(flavor: str, attn_backend: str = "flex", *, hf_config) -> ModelSpec:
    """Sized from the checkpoint's config.json, as Megatron is; ``flavor`` only names the run."""
    return ModelSpec(
        name="glm5_next",
        flavor=flavor,
        model=build_model_config(hf_config),
        parallelize_fn=parallelize_glm5_next,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Glm5NextStateDictAdapter,
    )
