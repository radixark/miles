"""Keep Transformer Engine FusedAdam under its NVTETensor handle cap.

FusedAdam passes every grad, param and moment tensor of a param group to one ``multi_tensor_applier`` call,
and TE's C++ binding wraps each of them in an NVTETensor handle taken from a fixed pool (20,321 handles in
TE 2.17). A MoE that keeps each local expert's weights as separate tensors overflows it: Qwen3.6-35B-A3B at
EP4 (64 local experts x 40 layers) needs ~25k handles in one call, and the optimizer step fails with
"Cannot allocate 1 new NVTETensors. Maximum number of tensors reached: 20321".

Adam updates each tensor on its own, and the FusedAdam kernels only read the noop flag, so applying the
same op to aligned slices of the lists gives bitwise-identical results.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# 8192 handles stay well under TE's 20,321 pool even with the 8 lists of the fp8 kernel, and leave room for
# handles other TE code holds at the same time. The C++ applier launches one kernel per few dozen tensors
# anyway, so smaller calls add no kernel launches.
MAX_HANDLES_PER_CALL = 8192


def split_tensor_lists(tensor_lists: Sequence[Sequence[Any]], *, max_handles: int) -> list[list[list[Any]]]:
    """Slice parallel per-tensor lists into aligned chunks that each need at most ``max_handles`` handles."""
    num_tensors = len(tensor_lists[0])
    assert all(
        len(tensors) == num_tensors for tensors in tensor_lists
    ), f"multi-tensor lists must be aligned, got lengths {[len(tensors) for tensors in tensor_lists]}"
    tensors_per_chunk = max(1, max_handles // len(tensor_lists))
    return [
        [list(tensors[start : start + tensors_per_chunk]) for tensors in tensor_lists]
        for start in range(0, num_tensors, tensors_per_chunk)
    ]


class ChunkedMultiTensorApplier:
    """A ``multi_tensor_applier`` that splits calls needing more than ``max_handles`` NVTETensor handles."""

    def __init__(self, applier: Callable[..., Any], *, max_handles: int = MAX_HANDLES_PER_CALL) -> None:
        self.applier = applier
        self.max_handles = max_handles

    def __call__(self, op: Callable[..., Any], noop_flag_buffer: Any, tensor_lists: list[list[Any]], *args: Any):
        if sum(len(tensors) for tensors in tensor_lists) <= self.max_handles:
            return self.applier(op, noop_flag_buffer, tensor_lists, *args)
        for chunk in split_tensor_lists(tensor_lists, max_handles=self.max_handles):
            self.applier(op, noop_flag_buffer, chunk, *args)


def install_fused_adam_chunking() -> None:
    """Route TE FusedAdam's multi-tensor calls through ``ChunkedMultiTensorApplier``; idempotent."""
    try:
        # optional dependency: absent on builds that train without Transformer Engine
        from transformer_engine.pytorch.optimizers import fused_adam
    except ImportError:
        return

    if isinstance(fused_adam.multi_tensor_applier, ChunkedMultiTensorApplier):
        return
    fused_adam.multi_tensor_applier = ChunkedMultiTensorApplier(fused_adam.multi_tensor_applier)
    logger.info(f"TE FusedAdam multi-tensor calls are split at {MAX_HANDLES_PER_CALL} NVTETensor handles")
