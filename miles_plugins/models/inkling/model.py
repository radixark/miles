"""Inkling model assembly: TransformerConfig construction, layer/block specs, and the
GPTModel subclass + provider that miles' custom-model-provider path loads."""

from __future__ import annotations

import torch
import transformer_engine.pytorch as te
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.inkling.layers import (
    InklingDenseMLP,
    InklingMoELayer,
    InklingRouter,
    InklingSelfAttention,
    InklingSharedExperts,
)
from miles_plugins.models.inkling.options import inkling_opt


class InklingExtra:
    def __init__(self, t: dict):
        self.num_attention_heads = t["num_attention_heads"]
        self.num_key_value_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        self.swa_num_attention_heads = t["swa_num_attention_heads"]
        self.swa_num_key_value_heads = t["swa_num_key_value_heads"]
        self.swa_head_dim = t["swa_head_dim"]
        self.sliding_window_size = t["sliding_window_size"]
        self.d_rel = t["d_rel"]
        self.rel_extent = t.get("rel_extent", 1024)
        self.local_layer_ids = set(t["local_layer_ids"])
        self.sconv_kernel_size = t["sconv_kernel_size"]
        self.use_sconv = t["use_sconv"]
        self.use_embed_norm = t["use_embed_norm"]
        self.n_routed_experts = t["n_routed_experts"]
        self.num_experts_per_tok = t["num_experts_per_tok"]
        self.n_shared_experts = t["n_shared_experts"]
        self.route_scale = t["route_scale"]
        self.intermediate_size = t["intermediate_size"]
        self.hidden_size = t["hidden_size"]
        self.num_hidden_layers = t["num_hidden_layers"]
        self.vocab_size = t["vocab_size"]
        self.rms_norm_eps = t["rms_norm_eps"]
        self.dense_mlp_idx = int(t.get("dense_mlp_idx", 0))
        self.dense_intermediate_size = int(t.get("dense_intermediate_size", t["intermediate_size"]))
        self.logits_mup_width_multiplier = t.get("logits_mup_width_multiplier", None)
        self.route_norm = t.get("route_norm", None)
        self.use_global_scale = t.get("use_global_scale", False)


def build_inkling_config(
    text_cfg: dict,
    tp=1,
    ep=1,
    pp=1,
    bf16=True,
    sp=False,
    etp=1,
    cp=1,
    varlen=True,
    permute_fusion=False,
    fp32_residual=False,
    pp_first_stage_layers=None,
    pp_last_stage_layers=None,
) -> TransformerConfig:
    inter = text_cfg["intermediate_size"]
    ns = text_cfg["n_shared_experts"]
    denseI = int(text_cfg.get("dense_intermediate_size", inter))
    cfg = TransformerConfig(
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_layers=text_cfg["num_hidden_layers"],
        hidden_size=text_cfg["hidden_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_query_groups=text_cfg["num_key_value_heads"],
        kv_channels=text_cfg["head_dim"],
        ffn_hidden_size=denseI,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        num_layers_in_first_pipeline_stage=pp_first_stage_layers,
        num_layers_in_last_pipeline_stage=pp_last_stage_layers,
        expert_model_parallel_size=ep,
        sequence_parallel=sp,
        expert_tensor_parallel_size=etp,
        context_parallel_size=cp,
        variable_seq_lengths=varlen,
        fp32_residual_connection=fp32_residual,
        moe_permute_fusion=permute_fusion,
        num_moe_experts=text_cfg["n_routed_experts"],
        moe_router_topk=text_cfg["num_experts_per_tok"],
        moe_ffn_hidden_size=inter,
        moe_shared_expert_intermediate_size=inter if ns > 0 else None,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_bias_update_rate=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=text_cfg["rms_norm_eps"],
        qk_layernorm=True,
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        gated_linear_unit=True,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    cfg.inkling = InklingExtra(text_cfg)
    cfg.moe_activation_in_fp32 = True
    cfg.moe_combine_in_fp32 = True
    _didx = cfg.inkling.dense_mlp_idx
    cfg.moe_layer_freq = [0] * _didx + [1] * (cfg.num_layers - _didx)
    cfg.hetereogenous_dist_checkpoint = True
    return cfg


def get_inkling_layer_spec(config) -> ModuleSpec:
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=False
    )
    attn = spec.submodules.self_attention
    spec.submodules.self_attention = ModuleSpec(
        module=InklingSelfAttention, params={"attn_mask_type": AttnMaskType.causal}, submodules=attn.submodules
    )
    mlp = spec.submodules.mlp
    moe_subs = mlp.submodules
    moe_subs.router = InklingRouter
    moe_subs.shared_experts = ModuleSpec(module=InklingSharedExperts, submodules=moe_subs.shared_experts.submodules)
    spec.submodules.mlp = ModuleSpec(
        module=InklingMoELayer, params=getattr(mlp, "params", None) or {}, submodules=moe_subs
    )
    return spec


def get_inkling_dense_layer_spec(config) -> ModuleSpec:
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=False)
    attn = spec.submodules.self_attention
    spec.submodules.self_attention = ModuleSpec(
        module=InklingSelfAttention, params={"attn_mask_type": AttnMaskType.causal}, submodules=attn.submodules
    )
    mlp = spec.submodules.mlp
    spec.submodules.mlp = ModuleSpec(
        module=InklingDenseMLP, params=getattr(mlp, "params", None) or {}, submodules=mlp.submodules
    )
    return spec


def get_inkling_block_spec(config, vp_stage=None):
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, _get_block_submodules
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    base = _get_block_submodules(config, get_inkling_layer_spec(config), vp_stage)
    dense_idx = getattr(config.inkling, "dense_mlp_idx", 0)
    if dense_idx <= 0:
        return base
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    local = [
        (get_inkling_dense_layer_spec(config) if (offset + i) < dense_idx else get_inkling_layer_spec(config))
        for i in range(len(base.layer_specs))
    ]
    return TransformerBlockSubmodules(layer_specs=local, layer_norm=base.layer_norm)


def get_inkling_spec(args, config, vp_stage=None):
    """--spec entry for the miles standard provider path."""
    import json

    text_cfg = json.load(open(f"{args.hf_checkpoint}/config.json"))["text_config"]
    config.inkling = InklingExtra(text_cfg)
    config.moe_router_dtype = "fp32"
    config.moe_router_bias_update_rate = 0.0
    config.moe_router_topk_scaling_factor = None
    config.moe_activation_in_fp32 = True
    config.moe_combine_in_fp32 = True
    return get_inkling_layer_spec(config)


class InklingGPTModel(GPTModel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if getattr(self, "pre_process", False) and getattr(self.config.inkling, "use_embed_norm", False):
            emb = self.embedding
            emb.embed_norm = te.RMSNorm(
                self.config.hidden_size, eps=self.config.inkling.rms_norm_eps, params_dtype=self.config.params_dtype
            )
            _orig_emb_forward = emb.forward

            _fp32res = bool(getattr(self.config, "fp32_residual_connection", False))

            def _emb_forward(*a, _orig=_orig_emb_forward, _norm=emb.embed_norm, _fp32=_fp32res, **k):
                out = _orig(*a, **k)
                if _fp32:
                    h = out.float()
                    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + getattr(_norm, "eps", 1e-6))
                    return (h * _norm.weight.float()).bfloat16().float()
                return _norm(out)

            emb.forward = _emb_forward
        if getattr(self, "post_process", False) and bool(getattr(self.config, "fp32_residual_connection", False)):
            fln = getattr(self.decoder, "final_layernorm", None)
            if fln is not None:
                _orig_fln = fln.forward
                _pdt = self.config.params_dtype

                def _fln_forward(x, *a, _orig=_orig_fln, _dt=_pdt, **k):
                    return _orig(x.to(_dt) if x.dtype != _dt else x, *a, **k)

                fln.forward = _fln_forward
        mup = getattr(self.config.inkling, "logits_mup_width_multiplier", None)
        if getattr(self, "post_process", False) and mup:
            _ol = self.output_layer
            _orig_ol_forward = _ol.forward
            _mup = float(mup)

            def _ol_forward(x, *a, _orig=_orig_ol_forward, _m=_mup, **k):
                return _orig(x / _m, *a, **k)

            _ol.forward = _ol_forward


def inkling_model_provider(pre_process=True, post_process=True, vp_stage=None):
    import json

    from megatron.training import get_args

    args = get_args()
    if getattr(args, "context_parallel_size", 1) > 1:
        assert getattr(args, "allgather_cp", False), "Inkling CP requires --allgather-cp (zigzag CP not supported)"
    text_cfg = json.load(open(f"{args.hf_checkpoint}/config.json"))["text_config"]
    config = build_inkling_config(
        text_cfg,
        tp=args.tensor_model_parallel_size,
        ep=args.expert_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        bf16=args.bf16,
        sp=args.sequence_parallel,
        etp=getattr(args, "expert_tensor_parallel_size", 1) or 1,
        cp=getattr(args, "context_parallel_size", 1) or 1,
        varlen=getattr(args, "variable_seq_lengths", True),
        permute_fusion=getattr(args, "moe_permute_fusion", False),
        fp32_residual=getattr(args, "fp32_residual_connection", False),
        pp_first_stage_layers=getattr(args, "decoder_first_pipeline_num_layers", None),
        pp_last_stage_layers=getattr(args, "decoder_last_pipeline_num_layers", None),
    )
    model = InklingGPTModel(
        config=config,
        transformer_layer_spec=get_inkling_block_spec(config, vp_stage=vp_stage),
        vocab_size=text_cfg["vocab_size"],
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        vp_stage=vp_stage,
    )
    if inkling_opt("inkling_mm_towers"):
        from miles_plugins.models.inkling.mm_towers import wire_mm_towers

        wire_mm_towers(model, args.hf_checkpoint)
    return model
