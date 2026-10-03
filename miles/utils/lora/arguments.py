import logging
from typing import Any

from miles.utils.hf_utils.config import load_hf_config
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import (
    LORA_TARGET_GROUPS,
    exclude_hf_lora_targets,
    expand_hf_lora_targets,
    expand_packed_hf_lora_targets,
    get_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)
from miles.utils.lora.utils import is_lora_enabled, matches_lora_target, targets_expert_leaves
from miles_plugins.models.inkling.lora import resolve_inkling_adapter_targets
from miles_plugins.models.kimi_k3.lora import resolve_kimi_k3_adapter_targets

logger = logging.getLogger(__name__)


def validate_lora_args(args):
    validate_multi_lora_args(args)
    if not is_lora_enabled(args):
        return
    assert args.train_backend == "megatron", "LoRA injection is not implemented for FSDP; use --train-backend megatron"
    assert args.lora_rank > 0, "LoRA requires a positive --lora-rank, including when loading an adapter"
    hf_config = load_hf_config(args.hf_checkpoint)
    args.hf_lora_targets, args.lora_adapter_targets = _resolve_lora_targets(args, hf_config)

    # Training and serving must agree on shared-outer grouped-expert LoRA (expert_dim=1).
    if args.experts_shared_outer_loras and hasattr(
        args, "sglang_experts_shared_outer_loras"
    ):  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
        args.sglang_experts_shared_outer_loras = True
    assert args.experts_shared_outer_loras == bool(
        getattr(args, "sglang_experts_shared_outer_loras", args.experts_shared_outer_loras)
    ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
    if targets_expert_leaves(args.hf_lora_targets):
        logger.warning(
            "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
            "shared-outer" if args.experts_shared_outer_loras else "per-expert",
        )


def _resolve_lora_targets(args, hf_config):
    """Return final HF and adapter targets without changing the CLI selectors in `args`."""
    hf_mapping = HfWeightMapping.from_config(hf_config)
    hf_modules = sorted(name.removesuffix(".weight") for name in hf_mapping.parameter_names)
    targets = parse_lora_targets(args.target_modules)
    exclusions = parse_lora_targets(args.exclude_modules) or []
    explicit_targets = []
    if targets is not None:
        expanded = expand_packed_hf_lora_targets(targets, hf_modules)
        explicit_targets = [
            target for target in dict.fromkeys(targets + expanded) if target not in (*LORA_TARGET_GROUPS, "all-linear")
        ]
        targets = expanded
    if exclusions:
        conflicts = {
            module
            for module in hf_modules + explicit_targets + exclusions
            if any(matches_lora_target(module, target) for target in explicit_targets)
            and any(matches_lora_target(module, exclude) for exclude in exclusions)
        }
        assert not conflicts, f"Explicit LoRA targets overlap --exclude-modules: {sorted(conflicts)}"
    if targets is None and args.multi_lora:
        targets = list(LORA_TARGET_GROUPS)
    targets = resolve_hf_lora_targets(hf_config.to_dict(), target_modules=targets)
    targets = exclude_hf_lora_targets(targets, exclusions)

    if all(any(matches_lora_target(module, target) for module in hf_modules) for target in targets + exclusions):
        hf_targets = targets
        if exclusions:
            selected = [
                module for module in hf_modules if any(matches_lora_target(module, target) for target in targets)
            ]
            hf_targets = exclude_hf_lora_targets(selected, exclusions)
    elif args.megatron_to_hf_mode == "bridge":
        # Only legacy Megatron selectors need Bridge before trainer creation.
        from miles.backends.megatron_utils.lora.target_modules import normalize_lora_targets_to_hf

        hf_targets = normalize_lora_targets_to_hf(
            args.hf_checkpoint,
            targets,
            canonical=args.lora_type == "canonical_lora",
            exclude_modules=exclusions,
            explicit_targets=explicit_targets,
        )
    else:
        layout = get_hf_lora_targets(hf_config.to_dict())
        hf_targets = exclude_hf_lora_targets(expand_hf_lora_targets(targets, layout), exclusions)

    adapter_targets = list(hf_targets)
    if args.megatron_to_hf_mode == "raw":
        if hf_config.model_type in ("inkling_model", "inkling_mm_model", "inkling_text"):
            adapter_targets = resolve_inkling_adapter_targets(hf_config.to_dict(), hf_targets)
        elif hf_config.model_type == "kimi_k3":
            adapter_targets = resolve_kimi_k3_adapter_targets(
                hf_targets,
                canonical=args.lora_type == "canonical_lora",
                experts_shared_outer_loras=args.experts_shared_outer_loras,
            )
    return hf_targets, adapter_targets


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate the trainer-side constraints of
    the slot machinery. A no-op for normal runs."""
    args.multi_lora = args.multi_lora_n_adapters > 0
    if not args.multi_lora:
        return

    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert args.context_parallel_size == 1, (
        "multi-LoRA requires --context-parallel-size 1: the Tinker losses zip "
        "full-length per-datum vectors against log_probs, which CP would shard"
    )
    assert args.pipeline_model_parallel_size == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
    )
    # Per-slot token spans assume sequence-major contiguous sample packing, which only 'thd' provides.
    assert args.qkv_format == "thd", (
        "Multi-LoRA requires --qkv-format thd: per-adapter token spans assume the "
        f"micro-batch packs samples contiguously, which bshd does not (got {args.qkv_format!r})."
    )
    assert not args.experts_shared_outer_loras, (
        "Multi-LoRA does not support --experts-shared-outer-loras; MoE expert adapters "
        "use the per-expert layout. Drop the flag (and --sglang-experts-shared-outer-loras)."
    )
    assert "muon" not in str(args.optimizer).lower(), (
        "Multi-LoRA does not support Muon: per-adapter decoupled stepping is only "
        "implemented for Adam-family per-slot optimizers"
    )
    assert not args.colocate, "Multi-LoRA requires separate training and sampling GPUs to retain accumulated gradients"
    assert (
        not args.indep_dp and "train" not in args.ft_components
    ), "Multi-LoRA does not support independent-DP training; remove 'train' from --ft-components"
    assert not args.offload_train, (
        "Multi-LoRA retains per-adapter gradient accumulation in GPU buffers between "
        "train calls; --offload-train would destroy it. Disable offload for multi-LoRA."
    )
    assert not args.enable_witness, (
        "Multi-LoRA runs without the distributed optimizer (per-slot LayerWise "
        "optimizers); the witness module assumes use_distributed_optimizer"
    )
    assert args.sglang_tokenizer_worker_num == 1, (
        "Multi-LoRA requires --sglang-tokenizer-worker-num 1: dynamic adapter loading "
        "requires a single tokenizer-side LoRA registry."
    )
    assert not args.calculate_per_token_loss, (
        "Multi-LoRA normalizes each sample by its adapter batch "
        "(sample-mean); per-token loss normalization would make adapter batch weights "
        "depend on batch contents. Drop --calculate-per-token-loss."
    )
    assert (args.optimizer or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot SlotOptimizer only "
        f"implements Adam semantics; got --optimizer {args.optimizer}"
    )
    args.megatron_to_hf_mode = "bridge"
