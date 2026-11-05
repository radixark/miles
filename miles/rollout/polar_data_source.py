"""Miles-side data-source wrapper for Polar-driven GRPO rollouts.

Port of ProRL-Agent-Server's ``slime_bridge.data_source`` adapted for the Miles
rollout package. This module exposes a ``RolloutDataSourceWithBuffer`` subclass
whose ``__len__`` is rounded up to the rollout batch size, so the rollout
``num_rollout_per_epoch = len(data_source) // rollout_batch_size`` floor
arithmetic never skips the trailing prompts of a dataset whose size is not a
multiple of the rollout batch size.

The public surface (class name and its ``__len__`` semantics) is kept
byte-for-byte identical to the Slime source. The class subclasses Miles' native
``RolloutDataSourceWithBuffer`` (``miles.rollout.data_source``). No ``polar``
package types are referenced at import time, so the module loads under a plain
Miles environment without Polar installed.
"""

from __future__ import annotations

import math

from miles.rollout.data_source import RolloutDataSourceWithBuffer

__all__ = [
    "ceil_to_batch_size",
    "CeilEpochRolloutDataSourceWithBuffer",
]


def ceil_to_batch_size(size: int, batch_size: int) -> int:
    """Round ``size`` up to a multiple of ``batch_size``.

    Returns ``0`` for a non-positive ``size`` and raises ``ValueError`` when
    ``batch_size`` is not positive.
    """
    if size <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return math.ceil(size / batch_size) * batch_size


class CeilEpochRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Expose a rounded-up epoch length for fixed-size rollout batches.

    Rollout computes ``num_rollout_per_epoch = len(data_source) // rollout_batch_size``.
    For datasets whose size is not divisible by the rollout batch size, the
    default floor behavior skips the tail prompts. Returning a rounded-up length
    lets the existing data source wrap only the final few prompts while still
    covering every prompt in the dataset once per epoch.
    """

    def __len__(self) -> int:
        # Miles' RolloutDataSource defines no __len__ of its own (unlike
        # Slime's), so the underlying dataset length is derived directly rather
        # than via super().__len__(). A None dataset means the source has no
        # prompts, matching Slime's zero-length behavior.
        if self.dataset is None:
            source_length = 0
        else:
            source_length = len(self.dataset)
        return ceil_to_batch_size(
            source_length,
            int(getattr(self.args, "rollout_batch_size", 1) or 1),
        )
