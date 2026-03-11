"""Tests for miles.utils.ft.controller.state_machines.recovery.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.recovery.models import (
    RECOVERY_STATE_TO_INT,
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryDoneSt,
    RecoveryState,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import EvictingSt, StoppingAndRestartingSt


class TestRecoveryStateConstruction:
    def test_realtime_checks_default(self) -> None:
        state = RealtimeChecksSt()
        assert state.pre_identified_bad_nodes == []

    def test_realtime_checks_with_bad_nodes(self) -> None:
        state = RealtimeChecksSt(pre_identified_bad_nodes=["node-0", "node-1"])
        assert state.pre_identified_bad_nodes == ["node-0", "node-1"]

    def test_stop_time_diagnostics(self) -> None:
        state = StopTimeDiagnosticsSt()
        assert isinstance(state, RecoveryState)

    def test_notify_humans(self) -> None:
        state = NotifyHumansSt(state_before="EvictingAndRestartingSt")
        assert state.state_before == "EvictingAndRestartingSt"

    def test_recovery_done(self) -> None:
        state = RecoveryDoneSt()
        assert isinstance(state, RecoveryState)


class TestRecoveryStateFrozen:
    def test_realtime_checks_frozen(self) -> None:
        state = RealtimeChecksSt()
        with pytest.raises(ValidationError):
            state.pre_identified_bad_nodes = ["node-x"]  # type: ignore[misc]


class TestEvictingAndRestartingFactories:
    def test_direct_restart(self) -> None:
        state = EvictingAndRestartingSt.direct_restart()

        assert isinstance(state.restart, StoppingAndRestartingSt)
        assert state.restart.bad_node_ids == []
        assert isinstance(state.failed_next_state, StopTimeDiagnosticsSt)

    def test_evict_and_restart(self) -> None:
        state = EvictingAndRestartingSt.evict_and_restart_next_stop_time_diag(bad_node_ids=["node-0", "node-1"])

        assert isinstance(state.restart, EvictingSt)
        assert state.restart.bad_node_ids == ["node-0", "node-1"]
        assert isinstance(state.failed_next_state, StopTimeDiagnosticsSt)

    def test_evict_and_restart_final(self) -> None:
        state = EvictingAndRestartingSt.evict_and_restart_final(bad_node_ids=["node-0"])

        assert isinstance(state.restart, EvictingSt)
        assert isinstance(state.failed_next_state, NotifyHumansSt)


class TestRecoveryStateToInt:
    def test_all_states_have_mapping(self) -> None:
        assert RealtimeChecksSt in RECOVERY_STATE_TO_INT
        assert EvictingAndRestartingSt in RECOVERY_STATE_TO_INT
        assert StopTimeDiagnosticsSt in RECOVERY_STATE_TO_INT
        assert NotifyHumansSt in RECOVERY_STATE_TO_INT
        assert RecoveryDoneSt in RECOVERY_STATE_TO_INT

    def test_values_are_monotonically_increasing(self) -> None:
        values = list(RECOVERY_STATE_TO_INT.values())
        assert values == sorted(values)
