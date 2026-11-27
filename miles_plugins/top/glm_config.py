"""Resolve GLM configuration and enforce implementation requirements once.

Tensor layout checks stay with operations.
"""

import math

import torch

from miles_plugins.top.fp8_config import checkpoint_is_block_fp8


def prepare_glm_config(args, config, *, contract):
    """Select bindings and resolve options before building model parameters."""
    from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

    if contract != TRUE_ON_POLICY_V1:
        raise NotImplementedError("[top] GLM requires the shared TOP MoE contract")
    fp8 = checkpoint_is_block_fp8(args.hf_checkpoint)
    if fp8 and getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        # The FP8 install fills the decoder, embedding, norm and head; an MTP block would keep its random init.
        raise NotImplementedError(
            "[top] GLM block-FP8 installs no trainer MTP block from the HF checkpoint; pass --mtp-num-layers 0 "
            "(serving's EAGLE draft is the checkpoint's own MTP layer)"
        )
    # Slot receipts require explicit, unfused permutation on the trainer side.
    config.moe_permute_fusion = False
    validate_glm_config(args, config)
    validate_glm_moe_config(config)
    from miles_plugins.top.glm_ops import TopGLMOps

    config.top_glm_ops = TopGLMOps(fp8=fp8)


def validate_glm_config(args, config):
    from miles.utils.hf_utils.config import load_hf_config

    hf = load_hf_config(args.hf_checkpoint)
    validate_rotary_config(config, hf)
    for field, hf_field in (
        ("hidden_size", "hidden_size"),
        ("num_moe_experts", "n_routed_experts"),
        ("moe_ffn_hidden_size", "moe_intermediate_size"),
        ("moe_router_topk", "num_experts_per_tok"),
        ("num_layers", "num_hidden_layers"),
        ("num_attention_heads", "num_attention_heads"),
        ("q_lora_rank", "q_lora_rank"),
        ("kv_lora_rank", "kv_lora_rank"),
        ("qk_head_dim", "qk_nope_head_dim"),
        ("v_head_dim", "v_head_dim"),
        ("qk_pos_emb_head_dim", "qk_rope_head_dim"),
    ):
        if getattr(config, field) != getattr(
            hf, hf_field
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise ValueError(f"[top] GLM checkpoint and trainer disagree on {field}")
    shared_width = hf.moe_intermediate_size * (
        getattr(hf, "n_shared_experts", 0) or 0
    )  # config-access-exempt: HF model families differ in optional n_shared_experts metadata
    if (config.moe_shared_expert_intermediate_size or 0) != shared_width:
        raise ValueError("[top] GLM shared-expert width differs from the checkpoint")
    if config.moe_router_topk_scaling_factor != hf.routed_scaling_factor:
        raise ValueError("[top] GLM routed scaling differs from the checkpoint")
    if hf.index_head_dim != 128 or hf.kv_lora_rank != 512 or hf.qk_rope_head_dim != 64:
        raise NotImplementedError("[top] GLM requires index head128 and MLA latent512+64")
    if (
        getattr(hf, "index_k_norm_type", "layer") != "layer"
    ):  # config-access-exempt: HF model families differ in optional index_k_norm_type metadata
        raise NotImplementedError("[top] GLM index keys require LayerNorm")
    if any(
        getattr(hf, name, 0) for name in ("index_init_tokens", "index_local_tokens")
    ):  # config-access-exempt: HF config field selected at runtime from a table of checked names
        raise NotImplementedError("[top] GLM forced prefix/local indices are not admitted")
    from sglang.srt.true_on_policy.glm import validate_index_layout

    # Serving honours indexer_types, cli_factor, index_topk_pattern and index_kpool; this trainer runs only the
    # index_topk_freq/offset rule.
    validate_index_layout(hf)
    if args.miles_dsa_topk_backend != "flashinfer":
        raise NotImplementedError("[top] GLM requires paired FlashInfer selection")
    if not checkpoint_is_block_fp8(args.hf_checkpoint):
        from sglang.srt.true_on_policy.glm import DSA_SELECTION_ENVS, validate_envs

        # BF16 selection reads serving's FlashInfer tie-break settings from this process.
        validate_envs(DSA_SELECTION_ENVS)


def validate_rotary_config(config, hf):
    """Match the unscaled full-dimension rotary operation shared by BF16 and FP8."""
    rope = (
        getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
    )  # config-access-exempt: HF model families differ in optional rope_parameters metadata
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if (
        rope_type != "default"
        or config.rotary_scaling_factor != 1.0
        or getattr(config, "rotary_percent", 1.0)
        != 1.0  # config-access-exempt: TransformerConfig versions and model families differ in optional rotary_percent
        or getattr(config, "seq_len_interpolation_factor", None) not in (None, 1.0)
    ):
        raise NotImplementedError("[top] GLM forward requires unscaled full-dimension native RoPE")
    base = rope.get(
        "rope_theta", getattr(hf, "rope_theta", None)
    )  # config-access-exempt: HF model families differ in optional rope_theta metadata
    if base is None or config.rotary_base != base:
        raise ValueError("[top] GLM rotary base must match the checkpoint")


def validate_glm_moe_config(config):
    required = {
        "params_dtype": torch.bfloat16,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "deepep",
        "moe_router_score_function": "sigmoid",
        "moe_router_pre_softmax": True,
        "moe_router_enable_expert_bias": True,
        "moe_router_fusion": False,
        "moe_permute_fusion": False,
        "moe_apply_probs_on_input": False,
        "add_bias_linear": False,
        "gated_linear_unit": True,
        "moe_shared_expert_overlap": False,
        "moe_shared_expert_gate": False,
    }
    for name, value in required.items():
        if (
            getattr(config, name, value) != value
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] GLM requires {name}={value!r}")
    if getattr(
        config, "moe_grouped_gemm", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_grouped_gemm
        # bind_moe_layer wraps SequentialMLP experts (per-expert parameters) behind the slot transport.
        raise NotImplementedError("[top] GLM runs SequentialMLP experts; drop --moe-grouped-gemm")
    for name in ("moe_expert_capacity_factor", "moe_latent_size", "moe_input_jitter_eps"):
        if (
            getattr(config, name, None) is not None
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] GLM does not support {name}")
    for name in ("moe_router_num_groups", "moe_router_group_topk"):
        if getattr(config, name, None) not in (
            None,
            1,
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError("[top] GLM supports single-group routing")
    if config.num_moe_experts % config.expert_model_parallel_size:
        raise NotImplementedError("[top] GLM EP must divide the checkpoint expert count")
    if config.hidden_size % 128 or config.moe_ffn_hidden_size % 128:
        raise NotImplementedError("[top] BF16 DeepGEMM requires 128-aligned expert dimensions")
    scaling = config.moe_router_topk_scaling_factor
    if scaling is None or not math.isfinite(scaling) or scaling <= 0:
        raise NotImplementedError("[top] GLM requires a finite positive routed scaling factor")
    refuse_ignored_router_losses(config)
    if (
        getattr(config, "moe_router_dtype", None) is not None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_dtype
        raise NotImplementedError(
            "[top] GLM routes at serving's precision (BF16 gate logits, FP32 sigmoid); drop --moe-router-dtype"
        )


def refuse_ignored_router_losses(config) -> None:
    """The TOP GLM router routes exactly as serving does and adds no loss term, so these would be silently ignored."""
    aux = getattr(
        config, "moe_aux_loss_coeff", 0.0
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_aux_loss_coeff
    if any(aux if isinstance(aux, (list, tuple)) else (aux,)):
        raise NotImplementedError("[top] GLM has no router aux loss; set --moe-aux-loss-coeff 0")
    if getattr(
        config, "moe_z_loss_coeff", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_z_loss_coeff
        raise NotImplementedError("[top] GLM has no router z-loss; drop --moe-z-loss-coeff")
    if getattr(
        config, "moe_router_force_load_balancing", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_force_load_balancing
        raise NotImplementedError("[top] GLM routes as serving does; drop --moe-router-force-load-balancing")
