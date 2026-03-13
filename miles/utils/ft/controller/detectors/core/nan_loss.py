from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.metrics import get_non_finite_loss
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


class NanLossDetector(BaseFaultDetector):
    """Detect non-finite (NaN / Inf) loss values and trigger recovery.

    This is the only loss-based detector in the chain. A separate
    "loss spike" detector (Paper §4.1: 5× increase triggers recovery) is
    intentionally omitted: under RL-GRPO training the policy loss
    fluctuates near zero, making ratio-based spike detection meaningless
    and prone to false positives.
    """

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        bad_loss = get_non_finite_loss(ctx.metric_store.mini_wandb)

        if bad_loss is not None:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"loss is {bad_loss}",
                trigger=TriggerType.NAN_LOSS,
            )

        return Decision.no_fault(reason="loss is normal")
