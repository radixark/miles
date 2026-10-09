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

__all__ = ["glm5_next_configs", "model_registry"]

_DIM = 4096
_VOCAB_SIZE = 154880
_NORM_EPS = 1e-5
_SWIGLU_LIMIT = 10.0
_NUM_EXPERTS = 288
_TOP_K = 8
_MOE_HIDDEN_DIM = 2048
_NUM_STREAMS = 4
_EXPERTS_INIT = {name: partial(nn.init.trunc_normal_, std=0.02) for name in ("w1_EFD", "w2_EDF", "w3_EFD")}


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _ffn(hidden_dim: int) -> ClampedFeedForward.Config:
    return ClampedFeedForward.Config(
        w1=_linear(_DIM, hidden_dim),
        w2=_linear(hidden_dim, _DIM),
        w3=_linear(_DIM, hidden_dim),
        swiglu_limit=_SWIGLU_LIMIT,
    )


def _moe():
    routed = make_routed_experts_config(
        dim=_DIM,
        hidden_dim=_MOE_HIDDEN_DIM,
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
        param_init=_EXPERTS_INIT,
        comm_backend="standard",
    )
    routed.inner_experts = ClampedGroupedExperts.Config(
        dim=_DIM,
        hidden_dim=_MOE_HIDDEN_DIM,
        num_experts=_NUM_EXPERTS,
        param_init=_EXPERTS_INIT,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    return make_moe_config(
        num_experts=_NUM_EXPERTS,
        router=make_router_config(
            dim=_DIM,
            num_experts=_NUM_EXPERTS,
            gate_param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
            top_k=_TOP_K,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
        ),
        routed_experts=routed,
        shared_experts=_ffn(_MOE_HIDDEN_DIM),
    )


def _kda() -> KimiDeltaAttention.Config:
    num_heads, head_dim = 64, 128
    proj = num_heads * head_dim
    return KimiDeltaAttention.Config(
        num_heads=num_heads,
        head_dim=head_dim,
        q_proj=_linear(_DIM, proj),
        k_proj=_linear(_DIM, proj),
        v_proj=_linear(_DIM, proj),
        q_conv1d=ShortConv.Config(channels=proj, kernel_size=4),
        k_conv1d=ShortConv.Config(channels=proj, kernel_size=4),
        v_conv1d=ShortConv.Config(channels=proj, kernel_size=4),
        b_proj=_linear(_DIM, num_heads),
        f_a_proj=_linear(_DIM, head_dim),
        f_b_proj=_linear(head_dim, proj),
        g_a_proj=_linear(_DIM, head_dim),
        g_b_proj=_linear(head_dim, proj),
        gate=KDAGate.Config(num_heads=num_heads, head_dim=head_dim, lower_bound=-5.0),
        o_norm=GatedRMSNorm.Config(dim=head_dim, eps=_NORM_EPS),
        o_proj=_linear(proj, _DIM),
    )


def _dsa() -> DSAAttention.Config:
    n_heads, q_lora_rank, kv_lora_rank, head_dim = 64, 1536, 512, 256
    index_heads, index_head_dim = 32, 128
    return DSAAttention.Config(
        n_heads=n_heads,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        wq_a=_linear(_DIM, q_lora_rank),
        q_norm=Glm5NextRMSNorm.Config(dim=q_lora_rank, eps=_NORM_EPS),
        wq_b=_linear(q_lora_rank, n_heads * head_dim),
        wkv_a=_linear(_DIM, kv_lora_rank),
        kv_norm=Glm5NextRMSNorm.Config(dim=kv_lora_rank, eps=_NORM_EPS),
        wkv_b=_linear(kv_lora_rank, n_heads * 2 * head_dim),
        o_proj=_linear(n_heads * head_dim, _DIM),
        indexer=KpoolIndexer.Config(
            dim=_DIM,
            num_heads=index_heads,
            head_dim=index_head_dim,
            topk=2048,
            kpool=4,
            wq_b=_linear(q_lora_rank, index_heads * index_head_dim),
            wk=_linear(_DIM, index_head_dim),
            k_norm=Glm5NextLayerNorm.Config(dim=index_head_dim, eps=1e-6),
            weights_proj=_linear(_DIM, index_heads),
        ),
    )


def _hyper_connection() -> HyperConnection.Config:
    return HyperConnection.Config(
        dim=_DIM, num_streams=_NUM_STREAMS, sinkhorn_iterations=20, eps=1e-6, norm_eps=_NORM_EPS
    )


def _block(*, linear_attention: bool, dense_mlp: bool) -> Glm5NextBlock.Config:
    return Glm5NextBlock.Config(
        hc_attn=_hyper_connection(),
        hc_ffn=_hyper_connection(),
        attention_norm=Glm5NextRMSNorm.Config(dim=_DIM, eps=_NORM_EPS),
        ffn_norm=Glm5NextRMSNorm.Config(dim=_DIM, eps=_NORM_EPS),
        kda=_kda() if linear_attention else None,
        attention=None if linear_attention else _dsa(),
        feed_forward=_ffn(12288) if dense_mlp else None,
        moe=None if dense_mlp else _moe(),
    )


def _model(*, full_attn_layers: set[int], num_layers: int, num_dense_layers: int) -> Glm5NextModel.Config:
    return Glm5NextModel.Config(
        dim=_DIM,
        vocab_size=_VOCAB_SIZE,
        num_streams=_NUM_STREAMS,
        max_position_embeddings=1048576,
        tok_embeddings=Embedding.Config(num_embeddings=_VOCAB_SIZE, embedding_dim=_DIM),
        norm=Glm5NextRMSNorm.Config(dim=_DIM, eps=_NORM_EPS),
        lm_head=_linear(_DIM, _VOCAB_SIZE),
        layers=[
            _block(linear_attention=i not in full_attn_layers, dense_mlp=i < num_dense_layers)
            for i in range(num_layers)
        ],
    )


glm5_next_configs = {
    "4layer": lambda: _model(full_attn_layers={1, 3}, num_layers=4, num_dense_layers=1),
    "flash": lambda: _model(full_attn_layers={i for i in range(45) if i % 4 == 3}, num_layers=45, num_dense_layers=3),
}


def model_registry(flavor: str, attn_backend: str = "varlen") -> ModelSpec:
    return ModelSpec(
        name="glm5_next",
        flavor=flavor,
        model=glm5_next_configs[flavor](),
        parallelize_fn=parallelize_glm5_next,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Glm5NextStateDictAdapter,
    )
