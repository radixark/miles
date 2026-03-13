from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.utils import inject_rollout_cell_alive, make_detector_context, make_fake_metric_store

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.core.rollout_crash import RolloutCrashDetector
from miles.utils.ft.controller.types import ActionType, TriggerType
from miles.utils.ft.utils.metric_names import ROLLOUT_CELL_ALIVE


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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=20 - i * 10))

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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10))

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0", "node-1"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == []
        assert "dead for 60s" in decision.reason

    def test_no_metric_with_active_nodes_returns_telemetry_blind(self) -> None:
        """When active nodes exist for a rollout cell but rollout_cell_alive
        metric is missing, previously returned no_fault, treating monitoring
        blindness as healthy. Now returns TELEMETRY_BLIND so humans investigate."""
        store = make_fake_metric_store()
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert decision.trigger == TriggerType.TELEMETRY_BLIND
        assert "rollout_cell_alive metric missing" in decision.reason

    def test_no_metric_without_active_nodes_returns_no_fault(self) -> None:
        """No active nodes and no metric → cell may not be running yet."""
        store = make_fake_metric_store()
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids=set(),
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "no rollout_cell_alive metric yet" in decision.reason

    def test_no_active_nodes_still_detects_crash(self) -> None:
        """Rollout crash is a cell-level software check. It used to be gated by
        active_node_ids, which meant missing or stale cell_node_ids mapping
        would silence the detector entirely even when rollout_cell_alive=0."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        for i in range(8):
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10))

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids=set(),
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.bad_node_ids == []

    def test_no_active_nodes_alive_cell_still_healthy(self) -> None:
        store = make_fake_metric_store()
        inject_rollout_cell_alive(store, cell_id="0", alive=True)
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids=set(),
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "cell alive" in decision.reason

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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10))

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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=40 - i * 10))

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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10))

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
            inject_rollout_cell_alive(store, cell_id="0", alive=False, timestamp=now - timedelta(seconds=70 - i * 10))

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


class TestRolloutCrashTelemetryBlind:
    """Verify that missing rollout_cell_alive metric is reported as
    TELEMETRY_BLIND when the cell has active nodes, rather than
    silently returning no_fault. Previously, the detector treated
    'no data' the same as 'healthy', masking metric pipeline failures."""

    def test_metric_exists_and_healthy_returns_no_fault(self) -> None:
        """With metric data present and cell alive, detector returns no_fault."""
        store = make_fake_metric_store()
        inject_rollout_cell_alive(store, cell_id="0", alive=True)
        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "cell alive" in decision.reason

    def test_deduplicator_id_includes_cell_id(self) -> None:
        """Telemetry blind notification should carry a dedup ID scoped to cell."""
        store = make_fake_metric_store()
        detector = RolloutCrashDetector(cell_id="my-cell", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "my-cell" in decision.notify_deduplicator_id


class TestRolloutCrashMultipleSeriesAggregation:
    def test_any_dead_series_treated_as_unhealthy(self) -> None:
        """When multiple series match the same cell_id (e.g. old + new exporter
        coexist during restarts), the detector previously used df['value'][0]
        which nondeterministically picked one series. Now it explicitly
        aggregates: if ANY series reports dead, the cell is treated as
        unhealthy."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Series A (exporter-old): alive=1
        store.ingest_samples(
            target_id="exporter-old",
            samples=[GaugeSample(name=ROLLOUT_CELL_ALIVE, labels={"cell_id": "0"}, value=1.0)],
            timestamp=now,
        )
        # Series B (exporter-new): alive=0
        store.ingest_samples(
            target_id="exporter-new",
            samples=[GaugeSample(name=ROLLOUT_CELL_ALIVE, labels={"cell_id": "0"}, value=0.0)],
            timestamp=now,
        )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)
        # Should NOT report "cell alive" — at least one series is dead
        assert "cell alive" not in decision.reason

    def test_all_series_alive_reports_healthy(self) -> None:
        """When all matching series report alive, the cell is healthy."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        store.ingest_samples(
            target_id="exporter-old",
            samples=[GaugeSample(name=ROLLOUT_CELL_ALIVE, labels={"cell_id": "0"}, value=1.0)],
            timestamp=now,
        )
        store.ingest_samples(
            target_id="exporter-new",
            samples=[GaugeSample(name=ROLLOUT_CELL_ALIVE, labels={"cell_id": "0"}, value=1.0)],
            timestamp=now,
        )

        detector = RolloutCrashDetector(cell_id="0", alive_threshold_seconds=60.0)
        ctx = make_detector_context(
            metric_store=store,
            active_node_ids={"node-0"},
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE
        assert "cell alive" in decision.reason
