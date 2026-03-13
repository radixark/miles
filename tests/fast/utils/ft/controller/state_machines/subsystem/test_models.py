"""Tests for miles.utils.ft.controller.state_machines.subsystem.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    NotifyDeduplicator,
    SubsystemState,
    RecoveringSt,
)
from miles.utils.ft.controller.state_machines.recovery.models import RealtimeChecksSt
from miles.utils.ft.controller.types import TriggerType


class TestSubsystemStateConstruction:
    def test_detecting_anomaly(self) -> None:
        state = DetectingAnomalySt()
        assert isinstance(state, SubsystemState)

    def test_recovering(self) -> None:
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.HANG,
            recovery_start_time=now,
        )

        assert state.trigger == TriggerType.HANG
        assert isinstance(state.recovery, RealtimeChecksSt)


class TestSubsystemStateFrozen:
    def test_detecting_anomaly_frozen(self) -> None:

        state = DetectingAnomalySt()

        with pytest.raises(ValidationError):
            state.model_config = {}  # type: ignore[misc]


class TestImmutableDefaults:
    """State model fields that hold collections must use immutable types
    (tuple, frozenset) rather than mutable defaults (list, set, dict) to
    prevent accidental mutation of shared default instances."""

    def test_known_bad_node_ids_is_tuple(self) -> None:
        from datetime import datetime, timezone

        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["a", "b"],
        )
        assert isinstance(state.known_bad_node_ids, tuple)
        assert state.known_bad_node_ids == ("a", "b")

    def test_known_bad_node_ids_default_is_empty_tuple(self) -> None:
        from datetime import datetime, timezone

        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        assert state.known_bad_node_ids == ()

    def test_pre_identified_bad_nodes_is_tuple(self) -> None:
        state = RealtimeChecksSt(pre_identified_bad_nodes=["x", "y"])
        assert isinstance(state.pre_identified_bad_nodes, tuple)
        assert state.pre_identified_bad_nodes == ("x", "y")

    def test_bad_node_ids_is_tuple(self) -> None:
        from miles.utils.ft.controller.state_machines.restart.models import EvictingSt

        state = EvictingSt(bad_node_ids=["a"])
        assert isinstance(state.bad_node_ids, tuple)
        assert state.bad_node_ids == ("a",)


class TestNotifyDeduplicator:
    def test_first_occurrence_allows_notify(self) -> None:
        dedup = NotifyDeduplicator()
        assert dedup.should_notify("hang:no_heartbeat") is True

    def test_repeated_occurrence_suppressed_after_sync(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.sync_active_ids({"hang:no_heartbeat"})
        assert dedup.should_notify("hang:no_heartbeat") is False

    def test_different_id_still_allows_notify(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.sync_active_ids({"hang:no_heartbeat"})
        assert dedup.should_notify("metric_blind:gpu") is True

    def test_cleared_id_allows_notify_again(self) -> None:
        """When a problem disappears (ID removed from active set),
        re-occurrence should trigger a new notification."""
        dedup = NotifyDeduplicator()
        dedup.sync_active_ids({"hang:no_heartbeat"})
        dedup.sync_active_ids(set())
        assert dedup.should_notify("hang:no_heartbeat") is True

    def test_none_dedup_id_always_allows(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.sync_active_ids({"some_other"})
        assert dedup.should_notify(None) is True
