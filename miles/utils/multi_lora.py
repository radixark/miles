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
    "define_new_adapter_metrics",
    "is_multi_lora_enabled",
    "make_rid",
    "min_groups_per_dp_split",
    "parse_adapter",
    "slot_lora_name",
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


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate and default the multi-LoRA arg
    surface. Called from ``miles_validate_args``; a no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    # Swap in the multi-LoRA rollout fn and data source unless the user pointed these flags elsewhere.
    standard_rollout_fns = (
        "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
        "miles.rollout.sglang_rollout.generate_rollout",
    )
    if args.rollout_function_path in standard_rollout_fns:
        args.rollout_function_path = "miles.rollout.multi_lora.async_rollout.generate_rollout_multi_lora"
    if args.data_source_path == "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        args.data_source_path = "miles.rollout.multi_lora.data_source.MultiLoRAAsyncDataSource"
    # The per-adapter data source is inherently global (the controller owns
    # what is sampleable); rollout workers must not shard it.
    args.rollout_global_dataset = True
    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.target_modules is not None, "--target-modules must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
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
    assert (getattr(args, "optimizer", "adam") or "adam").lower() in ("adam", "muon", "dist_muon"), (
        "Multi-LoRA per-slot optimizers support adam (AdamW) and muon: state isolation and "
        f"slot-retirement cleanup implement only those state layouts; got --optimizer {args.optimizer}"
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

    # Trainer DP size, used to validate adapter batch shapes; guarded for harnesses without megatron args set.
    if all(
        hasattr(args, name)
        for name in (
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        )
    ):
        from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp

        model_parallel = compute_megatron_world_size_except_dp(args)
        assert (
            args.world_size % model_parallel == 0
        ), f"actor world size {args.world_size} is not divisible by tp*pp*cp {model_parallel}"
        args.multi_lora_dp_size = args.world_size // model_parallel
    else:
        args.multi_lora_dp_size = None

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


def parse_adapter(rid: str) -> str:
    return rid.rsplit(RID_SEPARATOR, 1)[0]


def slot_lora_name(slot: int) -> str:
    """Engine-side LoRA adapter name for a controller slot. Weight pushes and
    every inference request (rollout and prefill scoring) must agree on this."""
    return f"__miles_slot_{slot}"


def min_groups_per_dp_split(n_samples_per_prompt: int, dp_size: int) -> int:
    """Minimum prompt-group count that splits cleanly across data-parallel
    ranks.

    Train batches only pop groups in multiples of this value, so each popped
    slice has a sample count divisible by ``dp_size`` with no trimming.

    Requires ``n_samples_per_prompt`` and ``dp_size`` to divide each other
    (one must be a multiple of the other).
    """
    larger = max(dp_size, n_samples_per_prompt)
    smaller = min(dp_size, n_samples_per_prompt)
    if larger % smaller == 0:
        return larger // n_samples_per_prompt
    raise ValueError(
        f"n_samples_per_prompt={n_samples_per_prompt} must be a divisor or a multiple of "
        f"the data-parallel size {dp_size} so whole prompt groups can split evenly across ranks"
    )
