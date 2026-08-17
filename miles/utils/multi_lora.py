"""Small multi-LoRA helpers shared across the rollout, trainer, and controller.

The controller-side machinery (AdapterRegistry, TinkerBackend,
TinkerHTTPServer) lives in ``miles/ray/tinker_backend/``.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RID_SEPARATOR",
    "is_multi_lora_enabled",
    "make_rid",
    "slot_lora_name",
    "targets_expert_leaves",
    "validate_multi_lora_args",
]


# Must not appear in adapter names so rid prefix aborts can't cross adapters.
RID_SEPARATOR = "::"


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


# Leaf module names that can live inside MoE experts (they also name the dense MLP
# projections); the bulk aliases expand to them during target-module resolution.
_EXPERT_LEAF_NAMES = frozenset({"linear_fc1", "linear_fc2", "gate_proj", "up_proj", "down_proj"})
_ALL_MODULE_ALIASES = frozenset({"all", "all-linear", "all_linear"})


def targets_expert_leaves(target_modules: Any) -> bool:
    """Whether ``target_modules`` can put adapters on MoE expert linears."""
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    entries = [str(tm).strip().lower() for tm in (target_modules or [])]
    if any(entry in _ALL_MODULE_ALIASES for entry in entries):
        return True
    # Map each entry (possibly a dotted or wildcard path) to its leaf module name.
    return any(entry.split(".")[-1] in _EXPERT_LEAF_NAMES for entry in entries)


def _recompute_source_recognizes_adapters(recompute_module: Any) -> bool:
    """Whether a bridge ``peft.recompute`` module's input-grad patch classifies
    multi-LoRA ``.adapters.<slot>.`` parameter names as adapter parameters.
    Source inspection, separated from the import so tests can probe real
    module files without touching the installed bridge."""
    import inspect

    try:
        source = inspect.getsource(recompute_module.maybe_enable_recompute_inputs_grad)
    except (AttributeError, OSError, TypeError):
        return False
    return ".adapters." in source


def _bridge_recompute_patch_recognizes_multi_lora() -> bool:
    """Whether the installed Megatron-Bridge can replay checkpointed regions
    grad-enabled for multi-LoRA.

    Adapter-only training leaves every layer input grad-free, so activation
    recompute only works because the bridge's PEFT patch
    (``megatron.bridge.peft.recompute.maybe_enable_recompute_inputs_grad``)
    forces TransformerBlock inputs to require grad when only adapters train.
    Bridges before radixark/Megatron-Bridge#27 (branch ``bridge`` @ 688d34b8)
    matched only single-LoRA ``.adapter.`` names, classified multi-LoRA
    ``.adapters.<slot>.`` params as trainable base weights, and skipped the
    patch — full recompute then silently zeroed every adapter gradient. An
    unimportable or unreadable bridge fails closed (treated as unfixed)."""
    try:
        from megatron.bridge.peft import recompute
    except Exception:
        return False
    return _recompute_source_recognizes_adapters(recompute)


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate and default the multi-LoRA arg
    surface. Called from ``miles_validate_args``; a no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    assert getattr(
        args, "tinker_backend", False
    ), "multi-LoRA now requires --tinker-backend: the dataset-driven adapter-sample-level path was removed"
    # The per-adapter data source is inherently global (the controller owns
    # what is sampleable); rollout workers must not shard it.
    args.rollout_global_dataset = True
    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.target_modules is not None, "--target-modules must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert getattr(args, "pipeline_model_parallel_size", 1) == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: no single rank holds a "
        "complete adapter to push to the rollout engines, and a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
    )
    # Activation recompute: a checkpointed region is only replayed grad-enabled
    # when its INPUT requires grad. Multi-LoRA trains adapter-only (frozen
    # base), so recompute shapes that checkpoint the adapters themselves —
    # 'full' granularity always, selective 'moe' when the expert leaves are
    # the targets — depend on the bridge's PEFT input-grad patch forcing
    # TransformerBlock inputs to require grad. On a bridge without the
    # multi-LoRA fix (radixark/Megatron-Bridge#27), no layer is ever replayed,
    # every adapter gradient is identically zero, and training is a silent
    # no-op under a truthful grad_norm=0.0 (reproduced: GPT-OSS 20B
    # expert-only LoRA, TP=2+SP, 4xH200, 2026-08-12). Refuse those shapes at
    # launch unless the installed bridge carries the fix.
    recompute_modules = list(getattr(args, "recompute_modules", None) or [])
    risky_full = getattr(args, "recompute_granularity", None) == "full"
    risky_moe = "moe" in recompute_modules and targets_expert_leaves(args.target_modules)
    if risky_full or risky_moe:
        bridge_fixed = _bridge_recompute_patch_recognizes_multi_lora()
        assert not risky_full or bridge_fixed, (
            "Multi-LoRA with --recompute-granularity full requires a Megatron-Bridge "
            "whose PEFT recompute patch recognizes multi-LoRA '.adapters.<slot>.' "
            "params (radixark/Megatron-Bridge#27, branch bridge @ 688d34b8). The "
            "installed bridge does not: maybe_enable_recompute_inputs_grad matches "
            "only single-LoRA '.adapter.' names, so the TransformerBlock input-grad "
            "hook is skipped, no checkpointed layer is ever replayed, and every "
            "adapter gradient is silently zero (grad_norm=0.0 on every step while "
            "the job keeps 'training'). Upgrade the bridge, or use "
            "--recompute-granularity selective (default recompute-modules core_attn; "
            "add moe_act for MoE activation memory)."
        )
        assert not risky_moe or bridge_fixed, (
            "Multi-LoRA with expert-module targets and 'moe' in --recompute-modules "
            "requires a Megatron-Bridge whose PEFT recompute patch recognizes "
            "multi-LoRA '.adapters.<slot>.' params (radixark/Megatron-Bridge#27, "
            "branch bridge @ 688d34b8): the checkpointed MoE region contains the "
            "expert adapters themselves, so with expert-only targets their replay "
            "depends entirely on the bridge's TransformerBlock input-grad hook — "
            "without it every adapter gradient is silently zero (grad_norm=0.0 on "
            "every step). Upgrade the bridge, or recompute the expert activation "
            "instead: --recompute-modules core_attn moe_act."
        )
    # Per-slot token spans assume sequence-major contiguous sample packing, which only 'thd' provides.
    assert getattr(args, "qkv_format", "thd") == "thd", (
        "Multi-LoRA requires --qkv-format thd: per-adapter token spans assume the "
        f"micro-batch packs samples contiguously, which bshd does not (got {args.qkv_format!r})."
    )
    assert not getattr(args, "experts_shared_outer_loras", False), (
        "Multi-LoRA does not support --experts-shared-outer-loras; MoE expert adapters "
        "use the per-expert layout. Drop the flag (and --sglang-experts-shared-outer-loras)."
    )
    # Expert-parallel sizes are checked post-finalize in _validate_multi_lora_moe_support:
    # --expert-tensor-parallel-size stays None until Megatron's own validate_args resolves it.
    assert "muon" not in str(getattr(args, "optimizer", "")).lower(), (
        "Multi-LoRA does not support Muon: per-adapter decoupled stepping is only "
        "implemented for Adam-family per-slot optimizers"
    )
    assert not args.colocate, (
        "Multi-LoRA requires disaggregated rollout engines: weight sync is only "
        "implemented for the distributed path, not the colocated tensor path."
    )
    assert (
        not getattr(args, "indep_dp", False) and "train" not in args.ft_components
    ), "Multi-LoRA does not support independent-DP training; remove 'train' from --ft-components"
    assert not args.offload_train, (
        "Multi-LoRA retains per-adapter gradient accumulation in GPU buffers between "
        "train calls; --offload-train would destroy it. Disable offload for multi-LoRA."
    )
    assert not getattr(args, "enable_witness", False), (
        "Multi-LoRA runs without the distributed optimizer (per-slot LayerWise "
        "optimizers); the witness module assumes use_distributed_optimizer"
    )
    assert getattr(args, "sglang_tokenizer_worker_num", 1) == 1, (
        "Multi-LoRA requires --sglang-tokenizer-worker-num 1: each tokenizer "
        "worker process holds its own LoRA registry, so per-step adapter "
        "upserts resolve against whichever worker the router picks and fail "
        "non-deterministically. sglang rejects the upsert at runtime anyway; "
        "fail at launch instead of burning GPU time until the first weight push."
    )
    assert not args.calculate_per_token_loss, (
        "Multi-LoRA normalizes each sample by its adapter batch "
        "(sample-mean); per-token loss normalization would make adapter batch weights "
        "depend on batch contents. Drop --calculate-per-token-loss."
    )
    assert (getattr(args, "optimizer", "adam") or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot optimizer isolation "
        "(slot optimizer construction, slot retirement state cleanup) only implements "
        f"Adam semantics; got --optimizer {args.optimizer}"
    )
    from miles.utils.environ import enable_experimental_ft_trainer

    assert not enable_experimental_ft_trainer(), (
        "Multi-LoRA is not supported with MILES_EXPERIMENTAL_FT_TRAINER=1: the v2 "
        "train group has no adapter reconcile verbs and does not return train outcomes"
    )
    # Batches are variable-sized; carry the exact sample
    # count through rollout conversion instead of trimming to --global-batch-size.
    assert not args.disable_rollout_trim_samples, (
        "Multi-LoRA computes the exact dynamic batch size in rollout postprocessing; "
        "do not pass --disable-rollout-trim-samples"
    )
    args.use_dynamic_global_batch_size = True
    args.megatron_to_hf_mode = "bridge"


def make_rid(adapter_name: str) -> str:
    return f"{adapter_name}{RID_SEPARATOR}{uuid.uuid4().hex}"


def slot_lora_name(slot: int) -> str:
    """Engine-side LoRA adapter name for a controller slot. Weight pushes and
    every inference request (rollout and prefill scoring) must agree on this."""
    return f"__miles_slot_{slot}"
