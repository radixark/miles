from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import Shard
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common import Embedding, Linear, Sigmoid
from torchtitan.models.common.config_utils import make_moe_config, make_routed_experts_config, make_router_config

from miles.backends.torchtitan_utils.models.inkling2.layers import (
    Inkling2Attention,
    Inkling2RMSNorm,
    MessageController,
    QKL2Clip,
    RelLogitsProj,
    ResidualQuery,
    ShortConv,
    SoftcappedSwiGLU,
)
from miles.backends.torchtitan_utils.models.inkling2.model import (
    Fp32OutLinear,
    Inkling2Block,
    Inkling2Model,
    LoopQueries,
)

__all__ = ["build_model_config", "dist_muon_layouts"]

_TRUNC_NORMAL_INIT = partial(nn.init.trunc_normal_, std=0.02)
_EXPERTS_INIT = {name: _TRUNC_NORMAL_INIT for name in ("w1_EFD", "w2_EDF", "w3_EFD")}
_V_NORM_EPS = 1e-6


@dataclass(frozen=True)
class _Architecture:
    """The looped Inkling2 text decoder, read from the checkpoint's ``text_config``."""

    dim: int
    num_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    norm_eps: float
    n_heads: int
    n_kv_heads: int
    head_dim: int
    swa_n_heads: int
    swa_n_kv_heads: int
    swa_head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    local_layer_ids: frozenset[int]
    sconv_kernel_size: int
    qk_clip: float
    rel_clip: float
    log_scaling_n_floor: int
    log_scaling_alpha: float
    block_size: int
    num_experts: int
    top_k: int
    moe_hidden_dim: int
    route_scale: float
    gate_cap: float
    up_cap: float
    final_logit_softcapping: float
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    max_context_length: int

    @classmethod
    def from_hf_config(cls, hf_config) -> "_Architecture":
        text = hf_config.text_config
        text = text if isinstance(text, dict) else text.to_dict()
        _require_supported(text)
        loop = text["loop_config"]
        return cls(
            dim=text["hidden_size"],
            num_layers=text["num_hidden_layers"],
            vocab_size=text["vocab_size"],
            unpadded_vocab_size=text["unpadded_vocab_size"],
            norm_eps=text["rms_norm_eps"],
            n_heads=text["num_attention_heads"],
            n_kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            swa_n_heads=text["swa_num_attention_heads"],
            swa_n_kv_heads=text["swa_num_key_value_heads"],
            swa_head_dim=text["swa_head_dim"],
            d_rel=text["d_rel"],
            rel_extent=text["rel_extent"],
            sliding_window=text["sliding_window_size"],
            local_layer_ids=frozenset(text["local_layer_ids"]),
            sconv_kernel_size=text["sconv_kernel_size"],
            qk_clip=text["qk_l2_clip_threshold"],
            rel_clip=text["rel_l2_clip_threshold"],
            log_scaling_n_floor=text["log_scaling_n_floor"],
            log_scaling_alpha=text["log_scaling_alpha"],
            block_size=text["n_layers_per_attn_res_block"],
            num_experts=text["n_routed_experts"],
            top_k=text["num_experts_per_tok"],
            moe_hidden_dim=text["intermediate_size"],
            route_scale=text["route_scale"],
            gate_cap=text["mlp_gate_softcap"],
            up_cap=text["mlp_up_softcap"],
            final_logit_softcapping=text["final_logit_softcapping"],
            core_start=loop["core_start"],
            core_end=loop["core_end"],
            num_passes=loop["num_passes"],
            gate_rank=loop["gate_rank"],
            max_context_length=text["model_max_length"],
        )


def _require_supported(text: dict) -> None:
    unsupported = {
        "n_shared_experts must be 0": text["n_shared_experts"] != 0,
        "use_gate_bias must be on": not text["use_gate_bias"],
        "attn_res_normalize_values must be on": not text["attn_res_normalize_values"],
        "use_value_norm_no_gamma must be on": not text["use_value_norm_no_gamma"],
        "q_bias / o_bias must be off": text["q_bias"] or text["o_bias"],
        "loop_config is required": "loop_config" not in text,
        "variable-pass loops (is_varloop) are not supported": text.get("loop_config", {}).get("is_varloop", False),
    }
    failed = [reason for reason, is_unsupported in unsupported.items() if is_unsupported]
    if failed:
        raise ValueError(f"inkling2 cannot build this checkpoint: {'; '.join(failed)}")


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _attention(arch: _Architecture, *, is_local: bool) -> Inkling2Attention.Config:
    n_heads = arch.swa_n_heads if is_local else arch.n_heads
    n_kv_heads = arch.swa_n_kv_heads if is_local else arch.n_kv_heads
    head_dim = arch.swa_head_dim if is_local else arch.head_dim
    kv_dim = n_kv_heads * head_dim
    sconv = ShortConv.Config(
        channels=kv_dim,
        kernel_size=arch.sconv_kernel_size,
        dynamic=not is_local,
        n_heads=n_kv_heads,
        dynamic_input_dim=arch.dim,
    )
    return Inkling2Attention.Config(
        dim=arch.dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_rel=arch.d_rel,
        is_local=is_local,
        sliding_window=arch.sliding_window,
        log_scaling_n_floor=arch.log_scaling_n_floor,
        log_scaling_alpha=arch.log_scaling_alpha,
        wq_du=_linear(arch.dim, n_heads * head_dim),
        wk_dv=_linear(arch.dim, kv_dim),
        wv_dv=_linear(arch.dim, kv_dim),
        wr_du=_linear(arch.dim, n_heads * arch.d_rel),
        gate_attn=_linear(arch.dim, n_heads * head_dim),
        attn_sink_proj=_linear(arch.dim, n_heads),
        wo_ud=_linear(n_heads * head_dim, arch.dim),
        k_sconv=sconv,
        v_sconv=sconv,
        q_l2_clip=QKL2Clip.Config(num_heads=n_heads, head_dim=head_dim, threshold=arch.qk_clip, learnable_scale=True),
        k_l2_clip=QKL2Clip.Config(
            num_heads=n_kv_heads, head_dim=head_dim, threshold=arch.qk_clip, learnable_scale=False
        ),
        rel_l2_clip=QKL2Clip.Config(
            num_heads=n_heads, head_dim=arch.d_rel, threshold=arch.rel_clip, learnable_scale=True
        ),
        rel_logits_proj=RelLogitsProj.Config(
            d_rel=arch.d_rel, rel_extent=arch.sliding_window if is_local else arch.rel_extent
        ),
        v_norm_eps=_V_NORM_EPS,
    )


def _moe(arch: _Architecture, *, enable_ep: bool):
    routed = make_routed_experts_config(
        dim=arch.dim,
        hidden_dim=arch.moe_hidden_dim,
        num_experts=arch.num_experts,
        top_k=arch.top_k,
        param_init=_EXPERTS_INIT,
        enable_ep=enable_ep,
    )
    routed.activation_fn = SoftcappedSwiGLU.Config(gate_cap=arch.gate_cap, up_cap=arch.up_cap)
    return make_moe_config(
        num_experts=arch.num_experts,
        router=make_router_config(
            dim=arch.dim,
            num_experts=arch.num_experts,
            gate_param_init={"weight": _TRUNC_NORMAL_INIT},
            top_k=arch.top_k,
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=arch.route_scale,
        ),
        routed_experts=routed,
        shared_experts=None,
        load_balance_coeff=1e-3,
        enable_sp=False,
    )


def _block(arch: _Architecture, layer_id: int, *, enable_ep: bool) -> Inkling2Block.Config:
    is_local = layer_id in arch.local_layer_ids
    sconv = ShortConv.Config(channels=arch.dim, kernel_size=arch.sconv_kernel_size)
    norm = Inkling2RMSNorm.Config(dim=arch.dim, eps=arch.norm_eps)
    in_core = arch.core_start <= layer_id < arch.core_end
    return Inkling2Block.Config(
        layer_id=layer_id,
        is_local=is_local,
        block_size=arch.block_size,
        attention=_attention(arch, is_local=is_local),
        input_norm=norm,
        mlp_norm=norm,
        feed_forward=_moe(arch, enable_ep=enable_ep),
        attn_sconv=sconv,
        mlp_sconv=sconv,
        attn_query=None if layer_id == 0 else ResidualQuery.Config(dim=arch.dim, count=1 if is_local else 3),
        mlp_query=ResidualQuery.Config(dim=arch.dim),
        controller=MessageController.Config(dim=arch.dim, rank=arch.gate_rank) if in_core else None,
    )


def build_model_config(
    flavor: str, *, parallelism: ParallelismConfig, seq_len: int, attn_backend: str = "flex", hf_config
) -> Inkling2Model.Config:
    """Sized from the checkpoint's config.json; ``flavor`` only names the run."""
    if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError("Inkling2 on torchtitan does not support tensor parallelism")
    if parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError("Inkling2 attention residuals do not support pipeline parallelism")
    if parallelism.context_parallel_degree > 1:
        raise NotImplementedError("Inkling2 on torchtitan does not support context parallelism yet")
    arch = _Architecture.from_hf_config(hf_config)
    if seq_len > arch.max_context_length:
        raise ValueError(f"--seq-length {seq_len} exceeds model_max_length {arch.max_context_length}")
    enable_ep = parallelism.expert_parallel_degree > 1
    norm = Inkling2RMSNorm.Config(dim=arch.dim, eps=arch.norm_eps)
    return Inkling2Model.Config(
        max_context_length=seq_len,
        dim=arch.dim,
        vocab_size=arch.vocab_size,
        tok_embeddings=Embedding.Config(num_embeddings=arch.vocab_size, embedding_dim=arch.dim),
        embed_norm=norm,
        norm=norm,
        lm_head=Fp32OutLinear.Config(in_features=arch.dim, out_features=arch.vocab_size),
        output_query=ResidualQuery.Config(dim=arch.dim),
        loop_queries=LoopQueries.Config(dim=arch.dim, num_core_layers=arch.core_end - arch.core_start),
        layers=[_block(arch, layer_id, enable_ep=enable_ep) for layer_id in range(arch.num_layers)],
        core_start=arch.core_start,
        core_end=arch.core_end,
        num_passes=arch.num_passes,
        unpadded_vocab_size=arch.unpadded_vocab_size,
        final_logit_softcapping=arch.final_logit_softcapping,
        log_scaling_n_floor=arch.log_scaling_n_floor,
        log_scaling_alpha=arch.log_scaling_alpha,
    )


_MUON_ATTENTION = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj", "wo_ud")
_MUON_CONTROLLER = ("down_weight", "gain_weight", "forget_weight")
_MUON_EXPERTS = ("w13.weight", "w2.weight")


def dist_muon_layouts(model: Inkling2Model.Config, parallelism: ParallelismConfig):
    """Which parameters DistMuon orthogonalizes and how: every dense projection whole on one owner rank,
    routed experts one matrix per expert in their EP/EDP storage. The rest (embeddings, lm_head, norms,
    convolutions, clip scales, relative-bias projections, residual queries) stays on AdamW."""
    owned = ComputeLayout(shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()}, compute_dtype=torch.bfloat16)
    if parallelism.expert_parallel_degree > 1:
        per_expert = ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.EDP_SHARD.value: Shard(0), MeshAxisName.EP.value: Shard(0)},
            shard_order_by_tensor_dim={0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value)},
            compute_dtype=torch.bfloat16,
        )
    else:
        per_expert = ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)}, compute_dtype=torch.bfloat16
        )

    layouts: dict[str, ComputeLayout] = {}
    buckets: list[BucketConfig] = []
    for layer_id, layer in enumerate(model.layers):
        prefix = f"layers.{layer_id}"
        dense = {f"{prefix}.attn.{name}.weight": owned for name in _MUON_ATTENTION}
        dense[f"{prefix}.feed_forward.router.gate.weight"] = owned
        if layer.controller is not None:
            for module in ("attn", "k", "v", "mlp"):
                dense.update({f"{prefix}.controllers.{module}.{name}": owned for name in _MUON_CONTROLLER})
        experts = {f"{prefix}.feed_forward.routed_experts.{name}": per_expert for name in _MUON_EXPERTS}
        layouts.update(dense)
        layouts.update(experts)
        buckets.append(BucketConfig(name=prefix, patterns=tuple(dense)))
        buckets.append(BucketConfig(name=f"{prefix}.routed-experts", patterns=tuple(experts)))
    pattern = (
        r"(?:"
        rf"attn\.(?:{'|'.join(_MUON_ATTENTION)})\.weight|"
        r"feed_forward\.router\.gate\.weight|"
        rf"controllers\.(?:attn|k|v|mlp)\.(?:{'|'.join(_MUON_CONTROLLER)})|"
        r"routed_experts\.(?:w13|w2)\.weight"
        r")$"
    )
    return pattern, layouts, tuple(buckets)
