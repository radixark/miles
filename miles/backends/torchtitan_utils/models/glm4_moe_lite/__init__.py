from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import ComplexRoPE, Embedding, Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3 import (
    _EMBEDDING_INIT,
    _NORM_INIT,
    DeepSeekV3Model,
    _build_dsv3_layers,
    _output_linear_init,
    parallelize_deepseekv3,
)
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = ["glm4_moe_lite_configs", "model_registry"]


def _30b_a3b(attn_backend: str) -> DeepSeekV3Model.Config:
    dim = 2048
    vocab_size = 154880
    rope_dim = 64
    layers = _build_dsv3_layers(
        n_layers=47,
        n_dense_layers=1,
        dim=dim,
        n_heads=20,
        q_lora_rank=768,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=rope_dim,
        v_head_dim=256,
        mscale=1.0,
        dense_hidden_dim=10240,
        moe_hidden_dim=1536,
        num_experts=64,
        num_shared_experts=1,
        router_top_k=4,
        router_score_func="sigmoid",
        router_route_scale=1.8,
        router_route_norm=True,
        attn_backend=attn_backend,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        rope=ComplexRoPE.Config(dim=rope_dim, max_seq_len=202752, theta=1_000_000.0),
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size, param_init=_output_linear_init(dim)),
        layers=layers,
        mtp_layers=[],
    )


glm4_moe_lite_configs = {"30B-A3B": _30b_a3b}


def model_registry(flavor: str, attn_backend: str = "flex") -> ModelSpec:
    return ModelSpec(
        name="glm4_moe_lite",
        flavor=flavor,
        model=glm4_moe_lite_configs[flavor](attn_backend=attn_backend),
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
