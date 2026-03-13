from __future__ import annotations

import logging
from datetime import timedelta

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.utils.metric_names import ROLLOUT_CELL_ALIVE
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType

logger = logging.getLogger(__name__)


class RolloutCrashDetector(BaseFaultDetector):
    """Detect rollout cell crash via rollout_cell_alive metric.

    Queries rollout_cell_alive{cell_id=X}. When alive=0 persists
    beyond alive_threshold_seconds, triggers restart with empty
    bad_node_ids (no eviction). Rollout crashes are software-level
    issues, not node hardware faults.
    """

    def __init__(
        self,
        *,
        cell_id: str,
        alive_threshold_seconds: float = 60.0,
    ) -> None:
        self._cell_id = cell_id
        self._threshold = alive_threshold_seconds

    def evaluate(self, ctx: DetectorContext) -> Decision:
        """Override base evaluate() to skip active_node_ids gating.

        Rollout crash detection is a cell-level software health check
        that only queries rollout_cell_alive{cell_id=X} and always
        returns bad_node_ids=[]. It should not be silenced when
        cell_node_ids mapping is missing or stale.
        """
        return self._evaluate_raw(ctx)

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        df = ctx.metric_store.time_series_store.query_latest(
            metric_name=ROLLOUT_CELL_ALIVE,
            label_filters={"cell_id": self._cell_id},
        )

        if df.is_empty():
            if ctx.active_node_ids:
                return Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=(
                        f"rollout_{self._cell_id}: rollout_cell_alive metric missing "
                        f"but cell has {len(ctx.active_node_ids)} active node(s)"
                    ),
                    trigger=TriggerType.TELEMETRY_BLIND,
                    notify_deduplicator_id=f"metric_blind:RolloutCrashDetector:{self._cell_id}",
                )
            return Decision.no_fault(
                reason=f"rollout_{self._cell_id}: no rollout_cell_alive metric yet"
            )

        # When multiple series match (e.g. old+new exporter coexist during
        # restarts), treat the cell as unhealthy if ANY series reports dead.
        # Only consider the cell alive when ALL matching series agree.
        if (df["value"] > 0).all():
            return Decision.no_fault(
                reason=f"rollout_{self._cell_id}: cell alive"
            )

        window = timedelta(seconds=self._threshold)
        range_df = ctx.metric_store.time_series_store.query_range(
            metric_name=ROLLOUT_CELL_ALIVE,
            window=window,
            label_filters={"cell_id": self._cell_id},
        )

        if range_df.is_empty():
            return Decision.no_fault(
                reason=f"rollout_{self._cell_id}: no range data"
            )

        time_span = (range_df["timestamp"].max() - range_df["timestamp"].min()).total_seconds()
        if time_span < self._threshold * 0.8:
            return Decision.no_fault(
                reason=f"rollout_{self._cell_id}: insufficient data span ({time_span:.1f}s < {self._threshold:.1f}s)"
            )

        all_dead = (range_df["value"] == 0).all()
        if not all_dead:
            return Decision.no_fault(
                reason=f"rollout_{self._cell_id}: cell intermittently dead, waiting"
            )

        logger.warning(
            "rollout_crash_detected cell_id=%s threshold=%.0f",
            self._cell_id,
            self._threshold,
        )
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=[],
            reason=f"rollout cell {self._cell_id} dead for {self._threshold:.0f}s",
            trigger=TriggerType.CRASH,
        )
