import logging
from datetime import datetime, timedelta, timezone

from miles.utils.ft.metric_names import DCGM_FI_DEV_GPU_TEMP
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models import ActionType, Decision
from miles.utils.ft.protocols.metrics import MetricQueryProtocol, TrainingMetricStoreProtocol

logger = logging.getLogger(__name__)

_DEFAULT_MFU_THRESHOLD_RATIO = 0.8
_DEFAULT_CONSECUTIVE_STEPS = 10
_DEFAULT_TEMPERATURE_DELTA_THRESHOLD = 20.0
_DEFAULT_DECLINE_TIMEOUT_MINUTES = 30.0
_DEFAULT_BASELINE_STEPS = 50


class MfuDeclineDetector(BaseFaultDetector):
    def __init__(
        self,
        mfu_baseline: float = 0.0,
        mfu_threshold_ratio: float = _DEFAULT_MFU_THRESHOLD_RATIO,
        consecutive_steps: int = _DEFAULT_CONSECUTIVE_STEPS,
        temperature_delta_threshold: float = _DEFAULT_TEMPERATURE_DELTA_THRESHOLD,
        decline_timeout_minutes: float = _DEFAULT_DECLINE_TIMEOUT_MINUTES,
        baseline_steps: int = _DEFAULT_BASELINE_STEPS,
        mfu_absolute_minimum: float = 0.0,
    ) -> None:
        if mfu_threshold_ratio <= 0 or mfu_threshold_ratio > 1:
            raise ValueError(f"mfu_threshold_ratio must be in (0, 1], got {mfu_threshold_ratio}")
        if consecutive_steps < 1:
            raise ValueError(f"consecutive_steps must be >= 1, got {consecutive_steps}")
        if temperature_delta_threshold <= 0:
            raise ValueError(f"temperature_delta_threshold must be > 0, got {temperature_delta_threshold}")
        if decline_timeout_minutes <= 0:
            raise ValueError(f"decline_timeout_minutes must be > 0, got {decline_timeout_minutes}")
        if baseline_steps < 1:
            raise ValueError(f"baseline_steps must be >= 1, got {baseline_steps}")
        if mfu_absolute_minimum < 0:
            raise ValueError(f"mfu_absolute_minimum must be >= 0, got {mfu_absolute_minimum}")

        self._mfu_baseline = mfu_baseline
        self._mfu_threshold_ratio = mfu_threshold_ratio
        self._consecutive_steps = consecutive_steps
        self._temperature_delta_threshold = temperature_delta_threshold
        self._decline_timeout_minutes = decline_timeout_minutes
        self._baseline_steps = baseline_steps
        self._mfu_absolute_minimum = mfu_absolute_minimum

        self._baseline_locked: bool = False
        self._locked_baseline: float | None = None

    def evaluate(self, ctx: DetectorContext) -> Decision:
        recent_mfu = ctx.mini_wandb.query_last_n_steps("mfu", last_n=self._consecutive_steps)
        if len(recent_mfu) < self._consecutive_steps:
            return Decision(action=ActionType.NONE, reason="insufficient MFU data")

        mfu_values = [value for _, value in recent_mfu]
        avg_mfu = sum(mfu_values) / len(mfu_values)

        if self._mfu_absolute_minimum > 0 and avg_mfu < self._mfu_absolute_minimum:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU {avg_mfu:.4f} below absolute minimum {self._mfu_absolute_minimum:.4f}",
            )

        baseline = self._get_baseline(ctx.mini_wandb)
        if baseline <= 0:
            return Decision(action=ActionType.NONE, reason="no valid MFU baseline")

        threshold = baseline * self._mfu_threshold_ratio
        mfu_stats = f"{avg_mfu:.4f} < {threshold:.4f}"

        if avg_mfu >= threshold:
            return Decision(action=ActionType.NONE, reason="MFU within acceptable range")

        high_temp_node = self._find_high_temperature_node(ctx.metric_store, ctx.rank_placement)
        if high_temp_node is not None:
            return Decision(
                action=ActionType.MARK_BAD_AND_RESTART,
                bad_node_ids=[high_temp_node],
                reason=f"MFU decline ({mfu_stats}) correlated with high temperature on {high_temp_node}",
            )

        elapsed_minutes = self._compute_decline_duration_minutes(ctx, threshold)

        if elapsed_minutes >= self._decline_timeout_minutes:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU decline ({mfu_stats}) persisted for {elapsed_minutes:.1f}min without identifiable cause",
            )

        return Decision(
            action=ActionType.NONE,
            reason=f"MFU declining ({mfu_stats}), monitoring ({elapsed_minutes:.1f}min)",
        )

    def _compute_decline_duration_minutes(
        self, ctx: DetectorContext, threshold: float,
    ) -> float:
        """Derive how long MFU has been below *threshold* from time-series data.

        Queries a window wider than the timeout so the "last healthy" reading
        is visible even when the decline started exactly at the timeout boundary.
        """
        lookup_window = timedelta(minutes=self._decline_timeout_minutes * 2)
        timed_mfu = ctx.mini_wandb.query_time_window(
            "mfu", window=lookup_window,
        )
        if not timed_mfu:
            return 0.0

        now = datetime.now(timezone.utc)

        last_healthy_time: datetime | None = None
        for _, ts, value in timed_mfu:
            if value >= threshold:
                last_healthy_time = ts

        if last_healthy_time is not None:
            return (now - last_healthy_time).total_seconds() / 60

        return (now - timed_mfu[0].timestamp).total_seconds() / 60

    def reset_baseline(self) -> None:
        """Clear the locked baseline so the next evaluate() recomputes it."""
        self._baseline_locked = False
        self._locked_baseline = None

    def _get_baseline(self, mini_wandb: TrainingMetricStoreProtocol) -> float:
        if self._mfu_baseline > 0:
            return self._mfu_baseline

        if self._baseline_locked and self._locked_baseline is not None:
            return self._locked_baseline

        total_needed = self._baseline_steps + self._consecutive_steps
        all_data = mini_wandb.query_last_n_steps("mfu", last_n=total_needed)

        baseline_data = all_data[:-self._consecutive_steps] if len(all_data) > self._consecutive_steps else []
        if not baseline_data:
            return 0.0

        baseline = sum(v for _, v in baseline_data) / len(baseline_data)

        self._locked_baseline = baseline
        self._baseline_locked = True
        logger.info("MFU baseline locked at %.4f from %d steps", baseline, len(baseline_data))

        return baseline

    def _find_high_temperature_node(
        self,
        metric_store: MetricQueryProtocol,
        rank_placement: dict[int, str],
    ) -> str | None:
        if not rank_placement:
            return None

        df = metric_store.query_latest(DCGM_FI_DEV_GPU_TEMP)
        if df.is_empty():
            return None

        node_ids = set(rank_placement.values())

        node_temps: dict[str, list[float]] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            if node_id in node_ids:
                node_temps.setdefault(node_id, []).append(row["value"])

        if not node_temps:
            return None

        node_avg_temps: dict[str, float] = {
            node_id: sum(temps) / len(temps)
            for node_id, temps in node_temps.items()
        }

        overall_avg = sum(node_avg_temps.values()) / len(node_avg_temps)

        for node_id, avg_temp in node_avg_temps.items():
            if avg_temp > overall_avg + self._temperature_delta_threshold:
                return node_id

        return None
