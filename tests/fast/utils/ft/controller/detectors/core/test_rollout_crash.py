from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.utils import (
    inject_rollout_cell_alive,
    make_detector_context,
    make_fake_metric_store,
)

from miles.utils.ft.controller.detectors.core.rollout_crash import RolloutCrashDetector
from miles.utils.ft.controller.types import ActionType, TriggerType


class TestRolloutCrashDetector:
    def test_cell_alive_returns_no_fault(self) -> None:
        store = make_fake_metric_store()
        inject_rollout_cell_alive(store, cell_id="0", alive=True)
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "cell alive" in decision.reason

    def test_cell_dead_insufficient_data_span_returns_no_fault(self) -> None:
        """Cell is dead but not enough time has passed to confirm persistent death."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Only 20s of data, threshold is 60s (need 80% = 48s)
        for i in range(3):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=20 - i * 10)
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "insufficient data span" in decision.reason

    def test_cell_dead_exceeding_threshold_enters_recovery(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # 70s of data with alive=False, threshold is 60s
        for i in range(8):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10)
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == []
        assert "dead for 60s" in decision.reason

    def test_no_metric_yet_returns_no_fault(self) -> None:
        store = make_fake_metric_store()
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "no rollout_cell_alive metric yet" in decision.reason

    def test_no_active_nodes_returns_no_fault(self) -> None:
        store = make_fake_metric_store()
        inject_rollout_cell_alive(store, cell_id="0", alive=False)
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids=set(),
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "no active nodes" in decision.reason

    def test_intermittent_dead_returns_no_fault(self) -> None:
        """Cell flips between alive and dead within the window -- not a persistent crash."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        for i in range(8):
            inject_rollout_cell_alive(
                store,
                cell_id="0",
                alive=(i % 2 == 0),
                timestamp=now - timedelta(seconds=70 - i * 10),
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "intermittently dead" in decision.reason

    def test_trigger_type_is_crash(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        for i in range(8):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10)
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.CRASH

    def test_boundary_at_exactly_80_percent_threshold_passes(self) -> None:
        """time_span == threshold * 0.8 should NOT be rejected as insufficient."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # threshold=50.0 → 80% boundary = 40.0s
        # Inject 5 samples spanning exactly 40s (all dead)
        for i in range(5):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=40 - i * 10)
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=50.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == []

    def test_rollout_crash_does_not_mark_nodes_as_bad(self) -> None:
        """Rollout crashes are software-level issues, not hardware faults.
        Previously all active nodes were marked as bad, causing unnecessary
        node eviction on rollout restart."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        for i in range(8):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10)
            )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1", "node-2"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == []
        assert decision.trigger == TriggerType.CRASH

    def test_different_cell_ids_are_isolated(self) -> None:
        """Dead data for cell '0' must not affect detector for cell '1'."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Cell "0": persistently dead for 70s
        for i in range(8):
            inject_rollout_cell_alive(
                store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10)
            )

        # Cell "1": alive
        inject_rollout_cell_alive(store, cell_id="1", alive=True)

        detector_0 = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        detector_1 = RolloutCrashDetector(cell_id="1", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision_0 = detector_0.evaluate(ctx)
        decision_1 = detector_1.evaluate(ctx)

        assert decision_0.action == ActionType.ENTER_RECOVERY
        assert decision_1.action == ActionType.NONE
        assert "cell alive" in decision_1.reason
