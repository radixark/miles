from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend.run_state import read_orchestrator_state
from miles.utils.pydantic_utils import FrozenStrictBaseModel

POLL_INTERVAL_SECONDS = 10.0

_DEAD_POD_PHASES = frozenset({"Failed", "Succeeded"})
_LOST_POD = "Deleted"

_NO_VERDICT_EXIT_CODE = 1


class RunOutcome(FrozenStrictBaseModel):
    exit_code: int
    reason: str


def wait_for_run(
    *,
    exit_file: str | Path,
    read_pod_phase: Callable[[], str | None],
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    log: Callable[[str], None] = print,
    min_generation: int = 0,
    stop: Callable[[], bool] | None = None,
) -> RunOutcome:
    while True:
        outcome = poll_once(exit_file=exit_file, read_pod_phase=read_pod_phase, min_generation=min_generation)
        if outcome is not None:
            log(f"[launcher] run finished: {outcome.reason} (exit code {outcome.exit_code})")
            return outcome
        if stop is not None and stop():
            return RunOutcome(exit_code=_NO_VERDICT_EXIT_CODE, reason="the launcher was asked to stop watching")
        sleep(poll_interval_seconds)


def poll_once(
    *, exit_file: str | Path, read_pod_phase: Callable[[], str | None], min_generation: int = 0
) -> RunOutcome | None:
    state = read_orchestrator_state(exit_file)
    if state is not None and state.generation < min_generation:
        state = None
    if state is not None and state.is_terminal:
        return RunOutcome(exit_code=state.exit_code or 0, reason="the orchestrator reported its exit code")

    phase = read_pod_phase()
    if phase is None:
        if state is None:
            return None
        return RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod is gone and left no exit code ({_LOST_POD})",
        )

    if phase in _DEAD_POD_PHASES:
        return RunOutcome(
            exit_code=_NO_VERDICT_EXIT_CODE,
            reason=f"the orchestrator pod reached {phase} without writing an exit code",
        )

    return None
