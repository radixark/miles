"""Tests for miles.utils.ft.controller.state_machines.subsystem.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.recovery.models import RealtimeChecksSt
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    NotifyDeduplicator,
    RecoveringSt,
    SubsystemState,
)
from miles.utils.ft.controller.types import Decision, TriggerType


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


def _decision_with_dedup_id(dedup_id: str | None) -> Decision:
    from miles.utils.ft.controller.types import ActionType, TriggerType

    return Decision(
        action=ActionType.NOTIFY_HUMAN,
        reason="test",
        trigger=TriggerType.MISC,
        notify_deduplicator_id=dedup_id,
    )


class TestNotifyDeduplicator:
    def test_first_occurrence_allows_notify(self) -> None:
        dedup = NotifyDeduplicator()
        result = dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        assert len(result) == 1

    def test_repeated_occurrence_suppressed(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        result = dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        assert len(result) == 0

    def test_different_id_still_allows_notify(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        result = dedup.check_batch([_decision_with_dedup_id("metric_blind:gpu")])
        assert len(result) == 1

    def test_cleared_id_allows_notify_again(self) -> None:
        """When a problem disappears (ID absent from next batch),
        re-occurrence should trigger a new notification."""
        dedup = NotifyDeduplicator()
        dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        dedup.check_batch([])
        result = dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        assert len(result) == 1

    def test_none_dedup_id_always_allows(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.check_batch([_decision_with_dedup_id("some_other")])
        result = dedup.check_batch([_decision_with_dedup_id(None)])
        assert len(result) == 1

    def test_duplicate_id_in_same_batch_both_allowed(self) -> None:
        """Two decisions with the same dedup_id in one batch: both pass
        because we check against the *previous* batch's active IDs, not
        the current batch being built."""
        dedup = NotifyDeduplicator()
        result = dedup.check_batch([
            _decision_with_dedup_id("hang:no_heartbeat"),
            _decision_with_dedup_id("hang:no_heartbeat"),
        ])
        assert len(result) == 2

    def test_mixed_none_and_non_none_in_batch(self) -> None:
        """Batch containing both None and non-None dedup IDs: None always
        passes, non-None passes if new."""
        dedup = NotifyDeduplicator()
        dedup.check_batch([_decision_with_dedup_id("hang:no_heartbeat")])
        result = dedup.check_batch([
            _decision_with_dedup_id(None),
            _decision_with_dedup_id("hang:no_heartbeat"),
            _decision_with_dedup_id("metric_blind:gpu"),
        ])
        assert len(result) == 2
        dedup_ids = [d.notify_deduplicator_id for d in result]
        assert None in dedup_ids
        assert "metric_blind:gpu" in dedup_ids

    def test_active_ids_reflects_latest_batch(self) -> None:
        dedup = NotifyDeduplicator()
        dedup.check_batch([
            _decision_with_dedup_id("a"),
            _decision_with_dedup_id("b"),
        ])
        assert dedup.active_ids == frozenset({"a", "b"})

        dedup.check_batch([_decision_with_dedup_id("b")])
        assert dedup.active_ids == frozenset({"b"})

        dedup.check_batch([])
        assert dedup.active_ids == frozenset()
