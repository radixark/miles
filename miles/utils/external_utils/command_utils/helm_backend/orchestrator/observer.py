from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import OrchestratorState
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 10.0

_DEAD_POD_PHASES = frozenset({"Failed", "Succeeded"})
_UNREADABLE_PHASE = "Unknown"
_MISSING_POD_POLLS = 3
_DEAD_POD_POLLS = 3
_FAILING_POD_POLLS = 3
_NO_VERDICT_EXIT_CODE = 1


class ObservedPod(FrozenStrictBaseModel):
    phase: str
    startup_failure: str | None = None


class _RunOutcome(FrozenStrictBaseModel):
    exit_code: int
    reason: str


class _GenerationRead(NamedTuple):
    readable: bool
    state_file: Path | None


def wait_for_run(
    *,
    state_file: str | Path,
    read_pod: Callable[[], ObservedPod | None],
    read_active_state_file: Callable[[], Path | None],
) -> _RunOutcome:
    state_file = Path(state_file)
    missing_polls = 0
    dead_polls = 0
    failing_polls = 0
    while True:
        generation = _read_active_generation(read_active_state_file)
        if not generation.readable:
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        if generation.state_file is not None and generation.state_file != state_file:
            logger.info(f"The active orchestrator generation moved from {state_file} to {generation.state_file}")
            state_file = generation.state_file
            missing_polls = 0
            dead_polls = 0
            failing_polls = 0

        try:
            observed = read_pod()
        except Exception:
            logger.warning("Could not read the orchestrator pod; retrying", exc_info=True)
            observed = ObservedPod(phase=_UNREADABLE_PHASE)
            dead_polls = 0
            failing_polls = 0
        else:
            missing_polls = missing_polls + 1 if observed is None else 0
            dead_polls = dead_polls + 1 if observed is not None and observed.phase in _DEAD_POD_PHASES else 0
            failing_polls = failing_polls + 1 if observed is not None and observed.startup_failure is not None else 0

        outcome = _compute_run_outcome(
            state=OrchestratorState.read(state_file),
            observed=observed,
            missing_polls=missing_polls,
            dead_polls=dead_polls,
            failing_polls=failing_polls,
        )
        if outcome is not None:
            recheck = _read_active_generation(read_active_state_file)
            if not recheck.readable:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue
            if recheck.state_file is not None and recheck.state_file != state_file:
                logger.info(
                    f"Discarding the verdict of {state_file}: a later orchestrator generation took the run over"
                )
                continue
            logger.info(f"Run finished: {outcome.reason} (exit code {outcome.exit_code})")
            return outcome
        time.sleep(_POLL_INTERVAL_SECONDS)


def _read_active_generation(read_active_state_file: Callable[[], Path | None]) -> _GenerationRead:
    try:
        return _GenerationRead(readable=True, state_file=read_active_state_file())
    except Exception:
        logger.warning("Could not read the active orchestrator generation; retrying", exc_info=True)
        return _GenerationRead(readable=False, state_file=None)


def _compute_run_outcome(
    *,
    state: OrchestratorState | None,
    observed: ObservedPod | None,
    missing_polls: int,
    dead_polls: int,
    failing_polls: int,
) -> _RunOutcome | None:
    if state is not None and state.is_terminal:
        if state.exit_code is None:
            return _RunOutcome(
                exit_code=_NO_VERDICT_EXIT_CODE, reason="the orchestrator reported a terminal state with no exit code"
            )
        return _RunOutcome(exit_code=state.exit_code, reason="the orchestrator reported its exit code")

    if observed is None:
        if state is None or missing_polls < _MISSING_POD_POLLS:
            return None
        return _RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod has been gone for {missing_polls} polls and left no exit code",
        )

    if observed.phase in _DEAD_POD_PHASES:
        if dead_polls < _DEAD_POD_POLLS:
            return None
        return _RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod reached {observed.phase} without writing an exit code",
        )

    if (failure := observed.startup_failure) is not None:
        if failing_polls < _FAILING_POD_POLLS:
            return None
        return _RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod's container has been {failure} for {failing_polls} polls and the "
            f"orchestrator wrote no exit code",
        )

    return None
