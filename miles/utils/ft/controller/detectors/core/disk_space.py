from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import check_disk_fault
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


class DiskSpaceLowDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        faults = check_disk_fault(ctx.metric_store.time_series_store)
        if not faults:
            return Decision.no_fault("disk space ok")

        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            bad_node_ids=[f.node_id for f in faults],
            reason="; ".join(f.reason for f in faults),
            trigger=TriggerType.MISC,
        )
