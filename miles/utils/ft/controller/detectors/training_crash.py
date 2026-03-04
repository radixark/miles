import polars as pl

from miles.utils.ft.controller.detectors._metric_names import TRAINING_JOB_STATUS
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, _get_non_finite_loss
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_JOB_STATUS_FAILED = -1.0


class TrainingCrashDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        df = metric_store.query_latest(TRAINING_JOB_STATUS)
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="training job not failed")

        failed = df.filter(pl.col("value") == _JOB_STATUS_FAILED)
        if failed.is_empty():
            return Decision(action=ActionType.NONE, reason="training job not failed")

        trigger = self._determine_trigger(mini_wandb)

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"training job failed (trigger={trigger})",
            trigger=trigger,
        )

    def _determine_trigger(self, mini_wandb: MiniWandb) -> str:
        if _get_non_finite_loss(mini_wandb) is not None:
            return "nan_loss"

        return "crash"
