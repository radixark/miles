import logging

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, check_metric_blind
from miles.utils.ft.controller.detectors.checks.hardware import check_disk_fault
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.metric_names import NODE_FILESYSTEM_AVAIL_BYTES

logger = logging.getLogger(__name__)


class DiskSpaceLowDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        blind = check_metric_blind(ctx, NODE_FILESYSTEM_AVAIL_BYTES, detector_name="DiskSpaceLowDetector")
        if blind is not None:
            return blind

        faults = check_disk_fault(ctx.metric_store.time_series_store)
        if not faults:
            return Decision.no_fault("disk space ok")

        logger.warning("detector: DiskSpaceLowDetector found faults on nodes=%s", [f.node_id for f in faults])
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            bad_node_ids=[f.node_id for f in faults],
            reason="; ".join(f.reason for f in faults),
            trigger=TriggerType.MISC,
        )
