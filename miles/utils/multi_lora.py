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
]


# Must not appear in adapter names so rid prefix aborts can't cross adapters.
RID_SEPARATOR = "::"


class EmptyBatchTimeoutError(RuntimeError):
    """No trainable groups arrived before empty-wait timeout."""


def is_multi_lora_enabled(args: Any) -> bool:
    return getattr(args, "multi_lora", False)


def define_new_adapter_metrics(snapshot: dict) -> None:
    """Declare metric axes for adapters not seen before ({name}/* ->
    {name}/step, {name}/perf/* -> rollout/step); already-declared adapters
    are skipped internally, so calling this every snapshot is free.

    Glob expansion only reaches one path segment, so {name}/perf/* keys are
    out of {name}/*'s reach despite the shared prefix — each key group must
    stay exactly one segment under its glob.

    Must run in the the primary tracking writer, whose wandb
    definitions are the only ones that reliably persist — and before the
    adapter's first metrics, which is guaranteed at snapshot time: an adapter
    can't ship step metrics until it has been promoted and trained a full
    adapter batch.
    """
    # lazy import tracking deps
    from miles.utils.tracking_utils.tracking import define_step_key_metric_group

    for name in {**snapshot["pending"], **snapshot["active"], **snapshot["retiring"]}:
        define_step_key_metric_group(prefix=name, step_key=f"{name}/step")
        define_step_key_metric_group(prefix=f"{name}/perf", step_key="rollout/step")


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
