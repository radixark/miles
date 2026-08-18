from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import OrchestratorState
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 10.0

_DEAD_POD_PHASES = frozenset({"Failed", "Succeeded"})
_UNREADABLE_PHASE = "Unknown"
_MISSING_POD_POLLS = 3
_DEAD_POD_POLLS = 3
_NO_VERDICT_EXIT_CODE = 1


class _RunOutcome(FrozenStrictBaseModel):
    exit_code: int
    reason: str


def wait_for_run(
    *,
    state_file: str | Path,
    read_pod_phase: Callable[[], str | None],
    read_active_state_file: Callable[[], Path | None],
) -> _RunOutcome:
    state_file = Path(state_file)
    missing_polls = 0
    dead_polls = 0
    while True:
        try:
            active_state_file = read_active_state_file()
        except Exception:
            logger.warning("Could not read the active orchestrator generation; retrying", exc_info=True)
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        if active_state_file is not None and active_state_file != state_file:
            logger.info(f"The active orchestrator generation moved from {state_file} to {active_state_file}")
            state_file = active_state_file
            missing_polls = 0
            dead_polls = 0

        try:
            phase = read_pod_phase()
        except Exception:
            logger.warning("Could not read the orchestrator pod's phase; retrying", exc_info=True)
            phase = _UNREADABLE_PHASE
            dead_polls = 0
        else:
            missing_polls = missing_polls + 1 if phase is None else 0
            dead_polls = dead_polls + 1 if phase in _DEAD_POD_PHASES else 0

        outcome = _compute_run_outcome(
            state=OrchestratorState.read(state_file),
            phase=phase,
            missing_polls=missing_polls,
            dead_polls=dead_polls,
        )
        if outcome is not None:
            logger.info(f"Run finished: {outcome.reason} (exit code {outcome.exit_code})")
            return outcome
        time.sleep(_POLL_INTERVAL_SECONDS)


def _compute_run_outcome(
    *, state: OrchestratorState | None, phase: str | None, missing_polls: int, dead_polls: int
) -> _RunOutcome | None:
    if state is not None and state.is_terminal:
        if state.exit_code is None:
            return _RunOutcome(
                exit_code=_NO_VERDICT_EXIT_CODE, reason="the orchestrator reported a terminal state with no exit code"
            )
        return _RunOutcome(exit_code=state.exit_code, reason="the orchestrator reported its exit code")

    if phase is None:
        if state is None or missing_polls < _MISSING_POD_POLLS:
            return None
        return _RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod has been gone for {missing_polls} polls and left no exit code",
        )

    if phase in _DEAD_POD_PHASES:
        if dead_polls < _DEAD_POD_POLLS:
            return None
        return _RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod reached {phase} without writing an exit code",
        )

    return None
