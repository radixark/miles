"""Tests for miles.utils.ft.controller.state_machines.recovery.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.recovery.models import (
    RECOVERY_STATE_TO_INT,
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    RecoveryState,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.state_machines.restart.models import Evicting, StoppingAndRestarting


class TestRecoveryStateConstruction:
    def test_realtime_checks_default(self) -> None:
        state = RealtimeChecks()
        assert state.pre_identified_bad_nodes == []

    def test_realtime_checks_with_bad_nodes(self) -> None:
        state = RealtimeChecks(pre_identified_bad_nodes=["node-0", "node-1"])
        assert state.pre_identified_bad_nodes == ["node-0", "node-1"]

    def test_stop_time_diagnostics(self) -> None:
        state = StopTimeDiagnostics()
        assert isinstance(state, RecoveryState)

    def test_notify_humans(self) -> None:
        state = NotifyHumans(state_before="EvictingAndRestarting")
        assert state.state_before == "EvictingAndRestarting"

    def test_recovery_done(self) -> None:
        state = RecoveryDone()
        assert isinstance(state, RecoveryState)


class TestRecoveryStateFrozen:
    def test_realtime_checks_frozen(self) -> None:
        state = RealtimeChecks()
        with pytest.raises(ValidationError):
            state.pre_identified_bad_nodes = ["node-x"]  # type: ignore[misc]


class TestEvictingAndRestartingFactories:
    def test_direct_restart(self) -> None:
        state = EvictingAndRestarting.direct_restart()

        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.bad_node_ids == []
        assert isinstance(state.failed_next_state, StopTimeDiagnostics)

    def test_evict_and_restart(self) -> None:
        state = EvictingAndRestarting.evict_and_restart(bad_node_ids=["node-0", "node-1"])

        assert isinstance(state.restart, Evicting)
        assert state.restart.bad_node_ids == ["node-0", "node-1"]
        assert isinstance(state.failed_next_state, StopTimeDiagnostics)

    def test_evict_and_restart_final(self) -> None:
        state = EvictingAndRestarting.evict_and_restart_final(bad_node_ids=["node-0"])

        assert isinstance(state.restart, Evicting)
        assert isinstance(state.failed_next_state, NotifyHumans)


class TestRecoveryStateToInt:
    def test_all_states_have_mapping(self) -> None:
        assert RealtimeChecks in RECOVERY_STATE_TO_INT
        assert EvictingAndRestarting in RECOVERY_STATE_TO_INT
        assert StopTimeDiagnostics in RECOVERY_STATE_TO_INT
        assert NotifyHumans in RECOVERY_STATE_TO_INT
        assert RecoveryDone in RECOVERY_STATE_TO_INT

    def test_values_are_monotonically_increasing(self) -> None:
        values = list(RECOVERY_STATE_TO_INT.values())
        assert values == sorted(values)
