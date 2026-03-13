import logging

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.metrics import get_non_finite_loss
from miles.utils.ft.controller.types import ActionType, Decision, TrainingMetricStoreProtocol, TriggerType

logger = logging.getLogger(__name__)


class TrainingCrashDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.FAILED:
            return Decision.no_fault(reason="training job not failed")

        trigger = self._determine_trigger(ctx.metric_store.mini_wandb)
        logger.info("detector: TrainingCrashDetector job failed, trigger=%s", trigger)

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"training job failed (trigger={trigger})",
            trigger=trigger,
        )

    def _determine_trigger(self, mini_wandb: TrainingMetricStoreProtocol) -> TriggerType:
        if get_non_finite_loss(mini_wandb) is not None:
            return TriggerType.NAN_LOSS

        return TriggerType.CRASH
