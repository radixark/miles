import math

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision


class NanLossDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        latest_loss = mini_wandb.latest("loss", rank=0)

        if latest_loss is not None and not math.isfinite(latest_loss):
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"loss is {latest_loss}",
                trigger="nan_loss",
            )

        return Decision(action=ActionType.NONE, reason="loss is normal")
