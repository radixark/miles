"""Tests for _find_restart_requestor and _update_external_execution_result."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from miles.utils.ft.controller.state_machines.main.handlers import (
    _find_restart_requestor,
    _update_external_execution_result,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomaly,
    Recovering,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestarting,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    ExternalExecutionResult,
    RestartingMainJob as RestartingMainJobRestart,
    StoppingAndRestarting,
)
from miles.utils.ft.controller.types import TriggerType


def _make_recovering_with_main_job_restart(
    *,
    external_execution_result: ExternalExecutionResult | None = None,
) -> Recovering:
    return Recovering(
        recovery=EvictingAndRestarting(
            restart=RestartingMainJobRestart(external_execution_result=external_execution_result),
            failed_next_state=StopTimeDiagnostics(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


def _make_recovering_with_evicting() -> Recovering:
    """Recovering whose restart is Evicting, not RestartingMainJobRestart."""
    return Recovering(
        recovery=EvictingAndRestarting(
            restart=Evicting(bad_node_ids=["node-1"]),
            failed_next_state=StopTimeDiagnostics(),
        ),
        trigger=TriggerType.HANG,
        recovery_start_time=datetime.now(timezone.utc),
    )


def _make_recovering_with_stopping_and_restarting() -> Recovering:
    """Recovering whose restart is StoppingAndRestarting, not RestartingMainJobRestart."""
    return Recovering(
        recovery=EvictingAndRestarting(
            restart=StoppingAndRestarting(bad_node_ids=[]),
            failed_next_state=StopTimeDiagnostics(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# _find_restart_requestor
# ---------------------------------------------------------------------------


class TestFindRestartRequestor:
    def test_returns_name_when_unfulfilled_restart_exists(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_main_job_restart(),
        }
        assert _find_restart_requestor(subsystems) == "gpu"

    def test_returns_none_for_empty_subsystems(self) -> None:
        assert _find_restart_requestor({}) is None

    def test_returns_none_when_all_detecting_anomaly(self) -> None:
        subsystems = {
            "gpu": DetectingAnomaly(),
            "net": DetectingAnomaly(),
        }
        assert _find_restart_requestor(subsystems) is None

    def test_returns_none_when_execution_result_is_set(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_main_job_restart(
                external_execution_result=ExternalExecutionResult.SUCCEEDED,
            ),
        }
        assert _find_restart_requestor(subsystems) is None

    def test_returns_none_when_restart_is_evicting(self) -> None:
        """Recovering → EvictingAndRestarting but restart is Evicting, not RestartingMainJob."""
        subsystems = {
            "gpu": _make_recovering_with_evicting(),
        }
        assert _find_restart_requestor(subsystems) is None

    def test_returns_none_when_restart_is_stopping_and_restarting(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_stopping_and_restarting(),
        }
        assert _find_restart_requestor(subsystems) is None

    def test_returns_first_matching_among_multiple_subsystems(self) -> None:
        """When multiple subsystems have unfulfilled restarts, return one of them."""
        subsystems = {
            "detecting": DetectingAnomaly(),
            "fulfilled": _make_recovering_with_main_job_restart(
                external_execution_result=ExternalExecutionResult.SUCCEEDED,
            ),
            "requestor": _make_recovering_with_main_job_restart(),
            "evicting": _make_recovering_with_evicting(),
        }
        assert _find_restart_requestor(subsystems) == "requestor"

    def test_skips_non_matching_and_finds_matching(self) -> None:
        """Mixed subsystem states — only the unfulfilled RestartingMainJob matches."""
        subsystems = {
            "a": DetectingAnomaly(),
            "b": _make_recovering_with_evicting(),
            "c": _make_recovering_with_main_job_restart(),
        }
        assert _find_restart_requestor(subsystems) == "c"


# ---------------------------------------------------------------------------
# _update_external_execution_result
# ---------------------------------------------------------------------------


class TestUpdateExternalExecutionResult:
    def test_sets_execution_result_to_succeeded(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = _update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert isinstance(result.recovery.restart, RestartingMainJobRestart)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED

    def test_preserves_other_fields(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = _update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, Recovering)
        assert result.trigger == state.trigger
        assert result.recovery_start_time == state.recovery_start_time
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert result.recovery.failed_next_state == state.recovery.failed_next_state
        assert result.recovery.restart.bad_node_ids == state.recovery.restart.bad_node_ids

    def test_does_not_mutate_original(self) -> None:
        """Frozen models — the original state must remain unchanged."""
        state = _make_recovering_with_main_job_restart()

        _update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(state.recovery, EvictingAndRestarting)
        assert isinstance(state.recovery.restart, RestartingMainJobRestart)
        assert state.recovery.restart.external_execution_result is None

    def test_sets_failed_result(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = _update_external_execution_result(state, ExternalExecutionResult.FAILED)

        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert isinstance(result.recovery.restart, RestartingMainJobRestart)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED

    def test_raises_on_detecting_anomaly_state(self) -> None:
        with pytest.raises(AssertionError, match="Unexpected state"):
            _update_external_execution_result(DetectingAnomaly(), ExternalExecutionResult.SUCCEEDED)

    def test_raises_on_recovering_with_non_matching_restart(self) -> None:
        """Recovering → EvictingAndRestarting but restart is Evicting, not RestartingMainJob."""
        state = _make_recovering_with_evicting()
        with pytest.raises(AssertionError, match="Unexpected state"):
            _update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

    def test_preserves_bad_node_ids(self) -> None:
        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=RestartingMainJobRestart(
                    bad_node_ids=["node-0", "node-1"],
                ),
                failed_next_state=StopTimeDiagnostics(),
            ),
            trigger=TriggerType.NAN_LOSS,
            recovery_start_time=datetime.now(timezone.utc),
        )

        result = _update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert isinstance(result.recovery.restart, RestartingMainJobRestart)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED
        assert result.recovery.restart.bad_node_ids == ["node-0", "node-1"]
