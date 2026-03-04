import math

from miles.utils.ft.controller.detectors._metric_names import TRAINING_JOB_STATUS
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_JOB_STATUS_FAILED = -1


class TrainingCrashDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        df = metric_store.instant_query(f"{TRAINING_JOB_STATUS} == {_JOB_STATUS_FAILED}")
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="training job not failed")

        trigger = self._determine_trigger(mini_wandb)

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"training job failed (trigger={trigger})",
            trigger=trigger,
        )

    def _determine_trigger(self, mini_wandb: MiniWandb) -> str:
        latest_loss = mini_wandb.latest("loss", rank=0)
        if latest_loss is not None and not math.isfinite(latest_loss):
            return "nan_loss"

        return "crash"
