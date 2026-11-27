"""DeepSeek-V4 TOP block spec: Megatron's mHC layer flow with roles bound to SGLang ops.

Builds the per-layer ModuleSpecs directly (Megatron's experimental-attention
builder requires Transformer Engine). Every layer is MoE; layers
1..moe_n_hash_layers route by token hash.
"""

import functools
import json
from pathlib import Path

from megatron.core.transformer.spec_utils import ModuleSpec

from miles_plugins.top.fp8_config import checkpoint_is_block_fp8


def dsv4_server_kwargs(args):
    """SGLang server args of the serving TOP program the trainer pins to."""
    return dict(
        kv_cache_dtype="fp8_e4m3",
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_dispatcher_output_dtype="bf16",
        disable_shared_experts_fusion=True,
    )


def validate_dsv4_config(config):
    required = {
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_enable_expert_bias": True,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "deepep",
        "moe_grouped_gemm": False,
    }
    for field, value in required.items():
        if (
            getattr(config, field, None) != value
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] DeepSeek-V4 requires {field}={value!r}")
    if config.moe_n_hash_layers <= 0:
        raise NotImplementedError("[top] DeepSeek-V4 requires moe_n_hash_layers")
    # MTP layers number from 1 and would read as hash layers; their install path does not exist either.
    if getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        raise NotImplementedError("[top] DeepSeek-V4 trainer does not support MTP layers")


def validate_dsv4_moe_config(config):
    """Megatron MoE options that TopDSV4Router and the slot-combine MoE layer would silently drop."""
    required = {
        "moe_permute_fusion": False,
        "moe_apply_probs_on_input": False,
        "moe_shared_expert_overlap": False,
        "moe_shared_expert_gate": False,
    }
    for name, value in required.items():
        if (
            getattr(config, name, value) != value
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] DeepSeek-V4 requires {name}={value!r}")
    for name in ("moe_expert_capacity_factor", "moe_input_jitter_eps", "moe_latent_size"):
        if (
            getattr(config, name, None) is not None
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] DeepSeek-V4 does not support {name}")
    # The TOP router routes exactly as serving does and adds no loss term.
    aux = getattr(
        config, "moe_aux_loss_coeff", 0.0
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_aux_loss_coeff
    if any(aux if isinstance(aux, (list, tuple)) else (aux,)):
        raise NotImplementedError("[top] DeepSeek-V4 has no router aux loss; set --moe-aux-loss-coeff 0")
    if getattr(
        config, "moe_z_loss_coeff", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_z_loss_coeff
        raise NotImplementedError("[top] DeepSeek-V4 has no router z-loss; drop --moe-z-loss-coeff")
    if getattr(
        config, "moe_router_force_load_balancing", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_force_load_balancing
        raise NotImplementedError("[top] DeepSeek-V4 routes as serving does; drop --moe-router-force-load-balancing")
    if (
        getattr(config, "moe_router_dtype", None) is not None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_dtype
        raise NotImplementedError(
            "[top] DeepSeek-V4 routes at serving's precision (FP32 gate logits); drop --moe-router-dtype"
        )


def validate_dsv4_checkpoint(args, config):
    """The trainer's shape and routing constants must be the HF checkpoint's; serving reads them from config.json."""
    from miles_plugins.top.dsv4_mhc import HC_EPS

    if not checkpoint_is_block_fp8(args.hf_checkpoint):
        raise NotImplementedError(
            "[top] DeepSeek-V4 installs its trainer weights from a 128x128 block-FP8 HF checkpoint; "
            f"{args.hf_checkpoint!r} is unquantized"
        )
    hf = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    for name, hf_name in (
        ("hidden_size", "hidden_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_moe_experts", "n_routed_experts"),
        ("moe_ffn_hidden_size", "moe_intermediate_size"),
        ("moe_router_topk", "num_experts_per_tok"),
        ("moe_router_topk_scaling_factor", "routed_scaling_factor"),
        ("moe_n_hash_layers", "num_hash_layers"),
        ("num_residual_streams", "hc_mult"),
        ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters"),
    ):
        if getattr(config, name, None) != hf.get(
            hf_name
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise ValueError(
                f"[top] DeepSeek-V4 checkpoint and trainer disagree on {name}: "
                f"{getattr(config, name, None)!r} vs {hf_name}={hf.get(hf_name)!r}"  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            )
    shared_width = hf["moe_intermediate_size"] * (hf.get("n_shared_experts") or 0)
    if (config.moe_shared_expert_intermediate_size or 0) != shared_width:
        raise ValueError("[top] DeepSeek-V4 shared-expert width differs from the checkpoint")
    if hf.get("hc_eps") != HC_EPS:
        raise NotImplementedError(f"[top] DeepSeek-V4 mHC requires hc_eps={HC_EPS}, got {hf.get('hc_eps')!r}")
    # A tied head would overwrite the installed embedding with the checkpoint's separate head.
    if hf.get("tie_word_embeddings") or not args.backend.untie_embeddings_and_output_weights:
        raise NotImplementedError(
            "[top] DeepSeek-V4 has an untied embedding and head; pass --untie-embeddings-and-output-weights"
        )


def dsv4_block(args, config, vp_stage, stamp, *, contract):
    """mHC streams replace the paired residual; the DS4 block owns its norms."""
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

    from miles_plugins.top.binding import enable_batch_invariant
    from miles_plugins.top.dsv4_attention import TopDSV4Attention
    from miles_plugins.top.dsv4_mhc import TopDSV4Layer, TopFinalRMSNorm, TopHCHead, TopHyperConnection, TopMhcNorm
    from miles_plugins.top.dsv4_moe import TopClampedSharedExpert, TopDSV4MoELayer, TopDSV4Router
    from miles_plugins.top.grouped_experts import build_top_grouped_mlp

    validate_dsv4_config(config)
    validate_dsv4_moe_config(config)
    validate_dsv4_checkpoint(args, config)
    _bind_sglang_tp_group()
    from miles_plugins.top.dsv4_export import register as register_export

    register_export()
    moe = get_moe_module_spec(use_te=False, num_experts=config.num_moe_experts, moe_grouped_gemm=False)
    sub = moe.keywords["submodules"]
    sub.router = TopDSV4Router
    sub.experts = functools.partial(
        build_top_grouped_mlp(), **getattr(sub.experts, "keywords", {})
    )  # config-access-exempt: the experts builder is a functools.partial only on some layer specs
    sub.shared_experts = functools.partial(TopClampedSharedExpert, **sub.shared_experts.keywords)
    mlp = functools.partial(TopDSV4MoELayer, **moe.keywords)
    layer = ModuleSpec(
        module=TopDSV4Layer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TopMhcNorm,
            self_attention=ModuleSpec(module=TopDSV4Attention),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=TopHyperConnection,
            pre_mlp_layernorm=TopMhcNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=TopHyperConnection,
        ),
    )
    stamp.extend(
        [
            ("mhc", "HyperConnectionModule", "SGLang mhc_pre(+norm)/mhc_post"),
            ("hc_head", "learned_output_contract", "SGLang fused_hc_head"),
            ("self_attention", "DeepSeekV4Attention", "SGLang DS4 attention program (private pool/backend)"),
            ("moe", "MoELayer", "SGLang sqrtsoftplus/hash gate, clamped FP8 experts, DeepEP slot combine"),
        ]
    )
    block = TransformerBlockSubmodules(
        layer_specs=[layer] * get_num_layers_to_build(config, vp_stage=vp_stage),
        layer_norm=TopFinalRMSNorm,
        hc_head_contraction=TopHCHead,
    )
    enable_batch_invariant()
    return block


def _bind_sglang_tp_group():
    """mHC reads SGLang's tp_group only to choose symmetric-memory allocation, which the trainer never uses."""
    from sglang.srt.runtime_context import get_parallel

    try:
        _ = get_parallel().tp_group
    except RuntimeError:
        get_parallel().override_permanently(tp_group=None)
