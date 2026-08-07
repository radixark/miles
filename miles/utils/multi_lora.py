"""Small multi-LoRA helpers shared across the rollout, trainer, and controller.

The controller-side machinery (AdapterRegistry, MultiLoRABackend,
MultiLoRAHTTPServer) lives in ``miles/ray/multi_lora/``.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EmptyBatchTimeoutError",
    "RID_SEPARATOR",
    "cache_extra_key",
    "define_new_adapter_metrics",
    "is_multi_lora_enabled",
    "make_rid",
    "parse_adapter",
    "rid_prefix",
    "serving_lora_name",
    "validate_multi_lora_args",
]


# Must not appear in adapter names so rid prefix aborts can't cross adapters.
RID_SEPARATOR = "::"


class EmptyBatchTimeoutError(RuntimeError):
    """No trainable groups arrived before empty-wait timeout."""


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


def define_new_adapter_metrics(snapshot: dict) -> None:
    """Declare metric axes for new adapters ({name}/* -> {name}/step, {name}/perf/* -> rollout/step); must run
    in the primary tracking writer. Already-declared adapters are skipped, so calling every snapshot is free."""
    # lazy import tracking deps
    from miles.utils.tracking_utils.tracking import define_step_key_metric_group

    for name in {**snapshot["pending"], **snapshot["active"], **snapshot["retiring"]}:
        define_step_key_metric_group(prefix=name, step_key=f"{name}/step")
        define_step_key_metric_group(prefix=f"{name}/perf", step_key="rollout/step")


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
    """Set ``args.multi_lora``, then validate and default the multi-LoRA arg
    surface. Called from ``miles_validate_args``; a no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    # Default to the multi-LoRA wrapper and its no-op manager-level data
    # source unless the user pointed these flags elsewhere.
    if args.rollout_function_path is None:
        args.rollout_function_path = "miles.rollout.multi_lora.rollout_fn.MultiLoRARolloutFn"
    if args.data_source_path == "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        args.data_source_path = "miles.rollout.multi_lora.data_source.MultiLoRANullDataSource"
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
    assert args.multi_lora_max_coalesce_wait_s >= 0, "--multi-lora-max-coalesce-wait-s must be non-negative"
    assert (getattr(args, "optimizer", "adam") or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot optimizer isolation "
        "(build_multi_lora_optimizer, slot retirement state cleanup) only implements "
        f"Adam semantics; got --optimizer {args.optimizer}"
    )
    from miles.utils.environ import enable_experimental_ft_trainer

    assert not enable_experimental_ft_trainer(), (
        "Multi-LoRA is not supported with MILES_EXPERIMENTAL_FT_TRAINER=1: the v2 "
        "train group has no reconcile_adapters and does not return train outcomes"
    )
    # --global-batch-size may legitimately be unset (Megatron derives it later);
    # leave the adapter cap unset too rather than multiplying None.
    if args.multi_lora_max_adapter_global_batch_size is None and getattr(args, "global_batch_size", None) is not None:
        args.multi_lora_max_adapter_global_batch_size = 4 * args.global_batch_size
    if args.multi_lora_max_adapter_global_batch_size is not None:
        assert (
            args.multi_lora_max_adapter_global_batch_size > 0
        ), "--multi-lora-max-adapter-global-batch-size must be positive"

    # Batch shaping rides the rollout-side DP schedule; the legacy dynamic-gbs
    # flag expects per-batch metadata multi-LoRA never emits, so fail at launch.
    assert not getattr(args, "use_dynamic_global_batch_size", False), (
        "Multi-LoRA does not take --use-dynamic-global-batch-size: variable batches "
        "ride the rollout-side DP schedule directly; drop the flag."
    )
    args.megatron_to_hf_mode = "bridge"


def make_rid(adapter_name: str, registration_id: str) -> str:
    """Request id carrying the full registration: a stale tenant's prefix abort
    can never match a same-name successor's requests."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}{uuid.uuid4().hex}"


def rid_prefix(adapter_name: str, registration_id: str) -> str:
    """Abort-by-prefix namespace for one registration of one adapter."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}"


def parse_adapter(rid: str) -> str:
    # The separator cannot appear in adapter names, so the first segment is the name.
    return rid.split(RID_SEPARATOR, 1)[0]


def serving_lora_name(adapter_name: str, registration_id: str) -> str:
    """Engine-side LoRA name for one registration; pushes and every inference
    request must agree on it, and a re-registered name is a new tenant."""
    return f"__miles_adapter_{adapter_name}_{registration_id}"


def cache_extra_key(adapter_name: str, registration_id: str, serving_version: int) -> str:
    """KV-cache namespace: registration and serving version both enter the key, so
    neither a re-registered name nor a republished revision can reuse stale KV."""
    return f"{adapter_name}:{registration_id}:v{serving_version}"
