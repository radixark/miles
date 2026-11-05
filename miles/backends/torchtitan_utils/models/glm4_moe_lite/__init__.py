from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, Sigmoid
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.models.deepseek_v3.flavors import _EMBEDDING_INIT, _NORM_INIT, _build_dsv3_layers, _output_linear_init

__all__ = ["MODEL_FLAVORS", "build_model_config"]

_RMS_NORM_EPS = 1e-5


def _use_hf_norm_eps_in_mla(config: DeepSeekV3Model.Config) -> None:
    for layer in config.layers:
        layer.attention.q_norm.eps = _RMS_NORM_EPS
        layer.attention.kv_norm.eps = _RMS_NORM_EPS


def _30b_a3b(*, attn_backend: str, seq_len: int, parallelism: ParallelismConfig) -> DeepSeekV3Model.Config:
    dim = 2048
    vocab_size = 154880
    rope_dim = 64
    enable_sp = parallelism.enable_sequence_parallel
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
        router_score_func=Sigmoid.Config(),
        router_route_scale=1.8,
        router_route_norm=True,
        attn_backend=attn_backend,
        rope=ComplexRoPE.Config(dim=rope_dim, max_context_length=seq_len, theta=1_000_000.0),
        norm_eps=_RMS_NORM_EPS,
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    config = DeepSeekV3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=_RMS_NORM_EPS,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=[],
    )
    _use_hf_norm_eps_in_mla(config)
    return config


MODEL_FLAVORS = {"30B-A3B": (_30b_a3b, 202752)}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> DeepSeekV3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(f"Requested seq_len {context_len} exceeds max context length {max_context_len} for {flavor}")
    return get_config(attn_backend=attn_backend, seq_len=context_len, parallelism=parallelism)
