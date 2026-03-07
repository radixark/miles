from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import _check_disk_fault
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType


class DiskSpaceLowDetector(BaseFaultDetector):
    is_critical = False

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = _check_disk_fault(ctx.metric_store)
        if not faults:
            return Decision.no_fault("disk space ok")

        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="; ".join(f.reason for f in faults),
            trigger=TriggerType.MISC,
        )
