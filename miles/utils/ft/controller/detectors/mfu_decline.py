from datetime import datetime, timezone

from miles.utils.ft.controller.detectors._metric_names import NODE_GPU_TEMPERATURE
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

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
    ) -> None:
        self._mfu_baseline = mfu_baseline
        self._mfu_threshold_ratio = mfu_threshold_ratio
        self._consecutive_steps = consecutive_steps
        self._temperature_delta_threshold = temperature_delta_threshold
        self._decline_timeout_minutes = decline_timeout_minutes

        self._dynamic_baseline: float | None = None
        self._decline_start_time: datetime | None = None

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        recent_mfu = mini_wandb.query_last_n_steps("mfu", rank=0, last_n=self._consecutive_steps)
        if len(recent_mfu) < self._consecutive_steps:
            self._decline_start_time = None
            return Decision(action=ActionType.NONE, reason="insufficient MFU data")

        baseline = self._get_baseline(mini_wandb)
        if baseline <= 0:
            return Decision(action=ActionType.NONE, reason="no valid MFU baseline")

        mfu_values = [value for _, value in recent_mfu]
        avg_mfu = sum(mfu_values) / len(mfu_values)
        threshold = baseline * self._mfu_threshold_ratio

        if avg_mfu >= threshold:
            self._decline_start_time = None
            return Decision(action=ActionType.NONE, reason="MFU within acceptable range")

        high_temp_node = self._find_high_temperature_node(metric_store, rank_placement)
        if high_temp_node is not None:
            self._decline_start_time = None
            return Decision(
                action=ActionType.MARK_BAD_AND_RESTART,
                bad_node_ids=[high_temp_node],
                reason=f"MFU decline ({avg_mfu:.4f} < {threshold:.4f}) correlated with high temperature on {high_temp_node}",
            )

        now = datetime.now(timezone.utc)
        if self._decline_start_time is None:
            self._decline_start_time = now

        elapsed_minutes = (now - self._decline_start_time).total_seconds() / 60
        if elapsed_minutes >= self._decline_timeout_minutes:
            self._decline_start_time = None
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU decline ({avg_mfu:.4f} < {threshold:.4f}) persisted for {elapsed_minutes:.1f}min without identifiable cause",
            )

        return Decision(
            action=ActionType.NONE,
            reason=f"MFU declining ({avg_mfu:.4f} < {threshold:.4f}), monitoring ({elapsed_minutes:.1f}min)",
        )

    def _get_baseline(self, mini_wandb: MiniWandb) -> float:
        if self._mfu_baseline > 0:
            return self._mfu_baseline

        if self._dynamic_baseline is not None:
            return self._dynamic_baseline

        baseline_data = mini_wandb.query_last_n_steps("mfu", rank=0, last_n=_DEFAULT_BASELINE_STEPS)
        if not baseline_data:
            return 0.0

        self._dynamic_baseline = sum(v for _, v in baseline_data) / len(baseline_data)
        return self._dynamic_baseline

    def _find_high_temperature_node(
        self,
        metric_store: MetricStoreProtocol,
        rank_placement: dict[int, str],
    ) -> str | None:
        if not rank_placement:
            return None

        df = metric_store.instant_query(NODE_GPU_TEMPERATURE)
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
