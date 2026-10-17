"""Qwen3.8-Flash-Next for the torchtitan backend: ``--titan-model-name qwen3_8_next``.

Flavors are the released checkpoint (``Flash-Next``, 48 layers) and its 4-layer
slice (``Flash-Next-4layer``, ``CharyZeng/Qwen3.8-Flash-Next-4layer``), which keeps
one of each layer kind: three GatedDeltaNet layers, one QSA layer, and the PLE layer.
"""

from functools import partial

from torch import nn
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.qwen3_5 import (
    _EMBEDDING_INIT,
    _LINEAR_INIT,
    _depth_experts_init,
    _depth_init,
    _offset_norm,
    _output_linear_init,
    _qwen35_deltanet_config,
    _shared_experts_config,
)
from torchtitan.models.qwen3_5.rope import MRoPE
from torchtitan.protocols.model_spec import ModelSpec

from miles.backends.torchtitan_utils.models.qwen3_8_next.model import (
    HeadContraction,
    HyperConnection,
    QSAAttention,
    Qwen38NextBlock,
    Qwen38NextModel,
    SigmoidGatedRMSNorm,
)
from miles.backends.torchtitan_utils.models.qwen3_8_next.ple import PerLayerEmbedding
from miles.backends.torchtitan_utils.models.qwen3_8_next.state_dict_adapter import Qwen38NextStateDictAdapter

__all__ = ["model_registry", "qwen3_8_next_configs"]

# text_config of Qwen/Qwen3.8-Flash-Next
_DIM = 2560
_VOCAB = 248320
_HC_COUNT = 4
_HC_LOWRANK = 320
_N_HEADS = 24
_N_KV_HEADS = 2
_HEAD_DIM = 256
_ROTARY_DIM = 64  # partial_rotary_factor 0.25
_ROPE_THETA = 10_000_000.0
_FULL_ATTENTION_INTERVAL = 4
_NUM_EXPERTS = 512
_TOP_K = 10
_EXPERT_HIDDEN = 640
_SHARED_EXPERT_HIDDEN = 640
_EOS_TOKEN_ID = 248044
# ple_layer_ids in the config is 1-based: [2] is decoder layer 1
_PLE_LAYERS = (1,)
_INDEXER = dict(
    indexer_n_heads=4,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_rope_theta=_ROPE_THETA,
)

_HC_INIT = {
    "hc_norm": nn.init.zeros_,
    "w_down": nn.init.xavier_uniform_,
    "w_up": nn.init.xavier_uniform_,
    "w_inject": nn.init.xavier_uniform_,
}
_PLE_INIT = {name: nn.init.zeros_ for name in ("norm_key", "norm_query", "norm_conv", "conv1d")}
_INDEX_NORM_INIT = {name: nn.init.zeros_ for name in ("index_q_norm", "index_k_norm")}


def _linear(in_features: int, out_features: int, param_init: dict = _LINEAR_INIT) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features, bias=False, param_init=param_init)


def _hyper_connection(*, inject: bool = True) -> HyperConnection.Config:
    config_cls = HyperConnection.Config if inject else HeadContraction.Config
    return config_cls(dim=_DIM, n=_HC_COUNT, lowrank=_HC_LOWRANK, param_init=_HC_INIT)


def _qsa_attention(layer_id: int) -> QSAAttention.Config:
    indexer_width = (_INDEXER["indexer_n_heads"] + 1) * _INDEXER["indexer_head_dim"]
    return QSAAttention.Config(
        n_heads=_N_HEADS,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        rotary_dim=_ROTARY_DIM,
        rope=MRoPE.Config(dim=_ROTARY_DIM, max_seq_len=262144, theta=_ROPE_THETA, mrope_section=[11, 11, 10]),
        wq=_linear(_DIM, _N_HEADS * _HEAD_DIM * 2),
        wk=_linear(_DIM, _N_KV_HEADS * _HEAD_DIM),
        wv=_linear(_DIM, _N_KV_HEADS * _HEAD_DIM),
        wo=_linear(_N_HEADS * _HEAD_DIM, _DIM, _depth_init(layer_id)),
        q_norm=_offset_norm(_HEAD_DIM),
        k_norm=_offset_norm(_HEAD_DIM),
        # only so the trainer builds varlen document offsets; QSA runs its own kernel
        inner_attention=get_attention_config("varlen"),
        index_qk_proj=_linear(_DIM, indexer_width),
        param_init=_INDEX_NORM_INIT,
        **_INDEXER,
    )


def _delta_net(layer_id: int):
    config = _qwen35_deltanet_config(
        dim=_DIM, n_key_heads=16, n_value_heads=48, key_head_dim=128, value_head_dim=128, layer_id=layer_id
    )
    config.norm = SigmoidGatedRMSNorm.Config(dim=128, eps=1e-6, param_init={"weight": nn.init.ones_})
    return config


def _moe(layer_id: int):
    return make_moe_config(
        num_experts=_NUM_EXPERTS,
        router=make_router_config(
            dim=_DIM,
            num_experts=_NUM_EXPERTS,
            gate_param_init=_depth_init(layer_id),
            top_k=_TOP_K,
            score_func="softmax",
            route_norm=True,
        ),
        routed_experts=make_routed_experts_config(
            dim=_DIM,
            hidden_dim=_EXPERT_HIDDEN,
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            param_init=_depth_experts_init(layer_id),
            comm_backend="standard",
        ),
        shared_experts=_shared_experts_config(dim=_DIM, hidden_dim=_SHARED_EXPERT_HIDDEN, layer_id=layer_id),
        # Megatron and sglang route with no expert bias; torchtitan's default would learn one
        load_balance_coeff=None,
    )


def _ple(layer_id: int) -> PerLayerEmbedding.Config:
    wide = _HC_COUNT * _DIM
    return PerLayerEmbedding.Config(
        dim=_DIM,
        n=_HC_COUNT,
        embed_dim=_DIM,
        hf_layer_index=layer_id,
        ngram_size=3,
        heads_per_ngram=8,
        conv_kernel_size=4,
        eos_token_id=_EOS_TOKEN_ID,
        key_proj=_linear(_DIM, wide),
        value_proj=_linear(_DIM, _DIM),
        param_init=_PLE_INIT,
    )


def _build(n_layers: int) -> Qwen38NextModel.Config:
    layers = []
    for layer_id in range(n_layers):
        full = (layer_id + 1) % _FULL_ATTENTION_INTERVAL == 0
        layers.append(
            Qwen38NextBlock.Config(
                attention=_qsa_attention(layer_id) if full else None,
                delta_net=None if full else _delta_net(layer_id),
                moe=_moe(layer_id),
                attn_hc=_hyper_connection(),
                mlp_hc=_hyper_connection(),
                ple=_ple(layer_id) if layer_id in _PLE_LAYERS else None,
            )
        )
    return Qwen38NextModel.Config(
        dim=_DIM,
        vocab_size=_VOCAB,
        hc_count=_HC_COUNT,
        tok_embeddings=Embedding.Config(num_embeddings=_VOCAB, embedding_dim=_DIM, param_init=_EMBEDDING_INIT),
        norm=_hyper_connection(inject=False),
        lm_head=Linear.Config(in_features=_DIM, out_features=_VOCAB, param_init=_output_linear_init(_DIM)),
        layers=layers,
    )


qwen3_8_next_configs = {
    "Flash-Next": partial(_build, n_layers=48),
    "Flash-Next-4layer": partial(_build, n_layers=4),
}


def parallelize_qwen3_8_next(
    model: Qwen38NextModel,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config: ActivationCheckpointingConfig | None,
    dump_folder: str,
):
    """FSDP, with routed experts on their own mesh under EP; TP, CP and PP are rejected by the model config."""
    from torchtitan.config import TORCH_DTYPE_MAP

    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("torch.compile is untested on Qwen3.8-Next's hyper-connection and QSA kernels")

    if parallel_dims.ep_enabled:
        # applies the expert sharding Qwen38NextModel.Config.update_from_config set before the build
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
    apply_fsdp_to_decoder(
        model,
        parallel_dims.get_mesh(dp_mesh_names),
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )
    return model


def model_registry(flavor: str, attn_backend: str = "flex") -> ModelSpec:
    # attn_backend is part of miles' registry call; QSA runs its own kernel whatever it says
    return ModelSpec(
        name="qwen3_8_next",
        flavor=flavor,
        model=qwen3_8_next_configs[flavor](),
        parallelize_fn=parallelize_qwen3_8_next,
        # pipeline parallelism is rejected in Qwen38NextModel.Config.update_from_config
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen38NextStateDictAdapter,
    )
