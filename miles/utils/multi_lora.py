"""Multi-LoRA arg surface and engine-side slot naming. Slot mechanics live in
``miles/backends/megatron_utils/lora/slots.py``."""

from dataclasses import dataclass
from typing import Any

__all__ = [
    "AdapterSpec",
    "is_multi_lora_enabled",
    "slot_lora_name",
    "targets_expert_leaves",
    "validate_multi_lora_args",
]


@dataclass(frozen=True)
class AdapterSpec:
    """What the weight-update path needs to export one slot's adapter."""

    slot: int
    rank: int
    alpha: float


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


def slot_lora_name(slot: int) -> str:
    """Engine-side LoRA adapter name for a slot. Weight pushes and every
    inference request must agree on this."""
    return f"__miles_slot_{slot}"


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


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate the trainer-side constraints of
    the slot machinery. A no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.target_modules is not None, "--target-modules must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert getattr(args, "pipeline_model_parallel_size", 1) == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: no single rank holds a "
        "complete adapter to push to the rollout engines, and a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
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
        "(build_multi_lora_optimizer, slot retirement state cleanup) only implements "
        f"Adam semantics; got --optimizer {args.optimizer}"
    )
    args.megatron_to_hf_mode = "bridge"
