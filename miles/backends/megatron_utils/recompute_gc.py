"""Collect reference cycles after each recomputed backward.

Full activation recomputation is supposed to keep one layer's activations alive at a time. It does
not, and the reason is the same one that affects the log-prob pass: an ``autograd.Function``'s
``ctx`` and its graph node reference each other, so refcounting cannot free either, and CPython's
generational collector triggers on *container allocation counts* -- a handful of multi-GiB tensors
barely move that counter.

Measured on a 4-layer DeepSeek-V4 at 128K with ``recompute_granularity=full``, bucketing the blocks
live at the memory peak by age: ~15 GiB had been allocated 10-60 s earlier, spanning two to ten
completed layer backwards. Tracing when they were finally released showed 12 GiB freed in one go
69.9 s *after* the peak -- not gradual reclamation, one outer boundary letting go at once.

Collecting explicitly at the end of each checkpointed backward moves that release to peak +0.6 s
and takes ~19 GiB off the peak at CP=8. Step time does not regress; the collections cost less than
moving the extra memory.

Off unless ``MILES_RECOMPUTE_BACKWARD_GC_GEN`` is set to a generation:

  * ``2`` (recommended) collects everything.
  * ``1`` measured bit-identical to 2, in peak and in step time.
  * ``0`` is WORSE THAN NOT COLLECTING. A cycle is only collectable when every object in it lies in
    a scanned generation, and surviving gen-0 objects get promoted; the tensors that were missed
    once because the cycle held one older object then leave ``collect(0)``'s view permanently.

This lives in Miles rather than in Megatron-LM because the image should not carry a patched
Megatron: the wrapper is applied to whatever ``megatron.core`` is installed. Doing it from the
outside is equivalent to doing it inside ``CheckpointFunction.backward`` -- the frame, and with it
every local naming an activation, is already gone by the time the wrapper regains control.
"""

from __future__ import annotations

import gc
import logging
import os

logger = logging.getLogger(__name__)

_ENV = "MILES_RECOMPUTE_BACKWARD_GC_GEN"
_installed = False


def recompute_backward_gc_generation() -> int:
    """Which GC generation to collect after a checkpointed backward; negative disables."""
    try:
        return int(os.environ.get(_ENV, "-1"))
    except ValueError:
        logger.warning(f"{_ENV} is not an integer; treating it as disabled")
        return -1


def enable_recompute_backward_gc() -> bool:
    """Wrap ``CheckpointFunction.backward`` to collect cycles when it returns.

    Idempotent, and a no-op unless the environment asks for it. Returns whether the wrapper is
    installed.
    """
    global _installed
    if _installed:
        return True

    generation = recompute_backward_gc_generation()
    if generation < 0:
        return False
    if generation == 0:
        logger.warning(
            f"{_ENV}=0 promotes surviving objects out of the only generation it scans, which "
            f"delays the collection it is trying to force. Use 1 or 2."
        )

    from megatron.core.tensor_parallel.random import CheckpointFunction

    # Accessing a staticmethod through the class yields the plain function.
    original = CheckpointFunction.backward

    def backward(ctx, *args):
        grads = original(ctx, *args)
        gc.collect(generation)
        return grads

    CheckpointFunction.backward = staticmethod(backward)
    _installed = True
    logger.info(f"recomputed backward will gc.collect({generation}) on return ({_ENV})")
    return True
