"""HACK ft-hang-repro: deterministic reproduction of the update_weights peer-death wedge.

This module exists ONLY to reproduce, on demand and every time, the bug where a cell that
dies inside ``update_weights`` leaves a surviving cell wedged. It is a throwaway debugging
aid; revert it via ``git grep 'HACK ft-hang-repro'`` once the root cause is understood.

Enable by exporting ``MILES_FT_HACK_KILL_AT_UPDATE_WEIGHTS=<N>``: on the Nth ``update_weights``
call on rank 0 of the active source cell, the process segfaults (matching the soak's segfault
fault mode), abruptly killing the cell mid-weight-update. ``update_weights`` only runs on the
first-alive cell, so N tracks that cell's rollouts.

``MILES_FT_HACK_KILL_PHASE`` picks WHERE in update_weights the death lands (default
``after_pause_before_gather``):
  - ``after_pause_before_gather``: engines paused, no weight transfer started yet. (r13: recovers.)
  - ``mid_engine_broadcast``: after the NCCL broadcast to engines is in flight, before it is
    waited -- leaves the engine-side weight-update group with a dead source.

The per-update_weights counter is bumped exactly once, at the ``after_pause_before_gather`` site
(which runs once per call, before any gather/broadcast), so it counts rollouts regardless of which
phase is selected to fire.
"""

import logging
import os

import torch.distributed as dist

from miles.utils.test_utils.fault_injector import inject_fault

logger = logging.getLogger(__name__)

_ENV_KILL_AT_UPDATE_WEIGHTS = "MILES_FT_HACK_KILL_AT_UPDATE_WEIGHTS"
_ENV_KILL_PHASE = "MILES_FT_HACK_KILL_PHASE"
_DEFAULT_PHASE = "after_pause_before_gather"
_COUNTER_PHASE = "after_pause_before_gather"

_update_weights_call_count = 0


def maybe_kill_at_update_weights(phase: str) -> None:
    global _update_weights_call_count

    raw_target = os.environ.get(_ENV_KILL_AT_UPDATE_WEIGHTS)
    if not raw_target:
        return
    if not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
        return

    if phase == _COUNTER_PHASE:
        _update_weights_call_count += 1

    selected_phase = os.environ.get(_ENV_KILL_PHASE, _DEFAULT_PHASE)
    if phase != selected_phase:
        return
    if _update_weights_call_count != int(raw_target):
        return

    logger.warning(
        "HACK ft-hang-repro: deterministic segfault at update_weights phase=%s call=%d pid=%d",
        phase,
        _update_weights_call_count,
        os.getpid(),
    )
    inject_fault("segfault")
