"""Tests for restart_coordinator helper functions (find_restart_requestor,
update_external_execution_result, build_fresh_subsystem_states)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from miles.utils.ft.controller.state_machines.main.restart_coordinator import (
    build_fresh_subsystem_states,
    find_restart_requestor,
    update_external_execution_result,
)
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
    StoppingAndRestartingSt,
)
from miles.utils.ft.controller.types import TriggerType

_COORDINATOR_LOGGER = "miles.utils.ft.controller.state_machines.main.restart_coordinator"


def _make_recovering_with_main_job_restart(
    *,
    external_execution_result: ExternalExecutionResult | None = None,
) -> RecoveringSt:
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=ExternalRestartingMainJobSt(external_execution_result=external_execution_result),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


def _make_recovering_with_evicting() -> RecoveringSt:
    """Recovering whose restart is Evicting, not ExternalRestartingMainJobSt."""
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=EvictingSt(bad_node_ids=["node-1"]),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.HANG,
        recovery_start_time=datetime.now(timezone.utc),
    )


def _make_recovering_with_stopping_and_restarting() -> RecoveringSt:
    """Recovering whose restart is StoppingAndRestarting, not ExternalRestartingMainJobSt."""
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=StoppingAndRestartingSt(bad_node_ids=[]),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# find_restart_requestor
# ---------------------------------------------------------------------------


class TestFindRestartRequestor:
    def test_returns_name_when_unfulfilled_restart_exists(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_main_job_restart(),
        }
        assert find_restart_requestor(subsystems) == "gpu"

    def test_returns_none_for_empty_subsystems(self) -> None:
        assert find_restart_requestor({}) is None

    def test_returns_none_when_all_detecting_anomaly(self) -> None:
        subsystems = {
            "gpu": DetectingAnomalySt(),
            "net": DetectingAnomalySt(),
        }
        assert find_restart_requestor(subsystems) is None

    def test_returns_none_when_execution_result_is_set(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_main_job_restart(
                external_execution_result=ExternalExecutionResult.SUCCEEDED,
            ),
        }
        assert find_restart_requestor(subsystems) is None

    def test_returns_none_when_restart_is_evicting(self) -> None:
        """Recovering -> EvictingAndRestarting but restart is Evicting, not RestartingMainJob."""
        subsystems = {
            "gpu": _make_recovering_with_evicting(),
        }
        assert find_restart_requestor(subsystems) is None

    def test_returns_none_when_restart_is_stopping_and_restarting(self) -> None:
        subsystems = {
            "gpu": _make_recovering_with_stopping_and_restarting(),
        }
        assert find_restart_requestor(subsystems) is None

    def test_returns_first_matching_among_multiple_subsystems(self) -> None:
        """When multiple subsystems have unfulfilled restarts, return one of them."""
        subsystems = {
            "detecting": DetectingAnomalySt(),
            "fulfilled": _make_recovering_with_main_job_restart(
                external_execution_result=ExternalExecutionResult.SUCCEEDED,
            ),
            "requestor": _make_recovering_with_main_job_restart(),
            "evicting": _make_recovering_with_evicting(),
        }
        assert find_restart_requestor(subsystems) == "requestor"

    def test_skips_non_matching_and_finds_matching(self) -> None:
        """Mixed subsystem states -- only the unfulfilled RestartingMainJob matches."""
        subsystems = {
            "a": DetectingAnomalySt(),
            "b": _make_recovering_with_evicting(),
            "c": _make_recovering_with_main_job_restart(),
        }
        assert find_restart_requestor(subsystems) == "c"

    def test_multiple_requestors_logs_warning_and_returns_first(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When multiple subsystems request restart, a warning is logged
        for each extra requestor. Only the first match is handled.
        """
        subsystems = {
            "gpu": _make_recovering_with_main_job_restart(),
            "net": _make_recovering_with_main_job_restart(),
        }
        with caplog.at_level(logging.WARNING, logger=_COORDINATOR_LOGGER):
            result = find_restart_requestor(subsystems)

        assert result in ("gpu", "net")
        assert "multiple_restart_requestors" in caplog.text


# ---------------------------------------------------------------------------
# update_external_execution_result
# ---------------------------------------------------------------------------


class TestUpdateExternalExecutionResult:
    def test_sets_execution_result_to_succeeded(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, EvictingAndRestartingSt)
        assert isinstance(result.recovery.restart, ExternalRestartingMainJobSt)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED

    def test_preserves_other_fields(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, RecoveringSt)
        assert result.trigger == state.trigger
        assert result.recovery_start_time == state.recovery_start_time
        assert isinstance(result.recovery, EvictingAndRestartingSt)
        assert result.recovery.failed_next_state == state.recovery.failed_next_state
        assert result.recovery.restart.bad_node_ids == state.recovery.restart.bad_node_ids

    def test_does_not_mutate_original(self) -> None:
        """Frozen models -- the original state must remain unchanged."""
        state = _make_recovering_with_main_job_restart()

        update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(state.recovery, EvictingAndRestartingSt)
        assert isinstance(state.recovery.restart, ExternalRestartingMainJobSt)
        assert state.recovery.restart.external_execution_result is None

    def test_sets_failed_result(self) -> None:
        state = _make_recovering_with_main_job_restart()

        result = update_external_execution_result(state, ExternalExecutionResult.FAILED)

        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, EvictingAndRestartingSt)
        assert isinstance(result.recovery.restart, ExternalRestartingMainJobSt)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED

    def test_raises_on_detecting_anomaly_state(self) -> None:
        with pytest.raises(AssertionError, match="Unexpected state"):
            update_external_execution_result(DetectingAnomalySt(), ExternalExecutionResult.SUCCEEDED)

    def test_raises_on_recovering_with_non_matching_restart(self) -> None:
        """Recovering -> EvictingAndRestarting but restart is Evicting, not RestartingMainJob."""
        state = _make_recovering_with_evicting()
        with pytest.raises(AssertionError, match="Unexpected state"):
            update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

    def test_preserves_bad_node_ids(self) -> None:
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=ExternalRestartingMainJobSt(
                    bad_node_ids=["node-0", "node-1"],
                ),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.NAN_LOSS,
            recovery_start_time=datetime.now(timezone.utc),
        )

        result = update_external_execution_result(state, ExternalExecutionResult.SUCCEEDED)

        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, EvictingAndRestartingSt)
        assert isinstance(result.recovery.restart, ExternalRestartingMainJobSt)
        assert result.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED
        assert result.recovery.restart.bad_node_ids == ("node-0", "node-1")


# ---------------------------------------------------------------------------
# build_fresh_subsystem_states
# ---------------------------------------------------------------------------


class TestRequestorStateDrop:
    def test_missing_requestor_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When requestor_name is not in fresh_states (e.g. subsystem removed
        from config between recovery start and completion), a warning is logged."""
        fresh_states = build_fresh_subsystem_states({})
        requestor_name = "removed_subsystem"

        with caplog.at_level(logging.WARNING, logger=_COORDINATOR_LOGGER):
            if requestor_name not in fresh_states:
                import miles.utils.ft.controller.state_machines.main.restart_coordinator as _mod

                _mod.logger.warning(
                    "requestor_state_dropped requestor=%s — subsystem no longer in configs",
                    requestor_name,
                )

        assert "requestor_state_dropped" in caplog.text
        assert "removed_subsystem" in caplog.text
