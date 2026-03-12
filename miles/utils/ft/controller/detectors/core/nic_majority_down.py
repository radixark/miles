from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import check_majority_nic_down
from miles.utils.ft.controller.types import Decision, TriggerType


class NicMajorityDownDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        faults = check_majority_nic_down(ctx.metric_store.time_series_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no majority NIC down",
            trigger=TriggerType.HARDWARE,
        )
