from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, check_metric_blind
from miles.utils.ft.controller.detectors.checks.hardware import check_majority_nic_down
from miles.utils.ft.controller.types import Decision, TriggerType
from miles.utils.ft.utils.metric_names import NODE_NETWORK_UP


class NicMajorityDownDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        blind = check_metric_blind(ctx, NODE_NETWORK_UP, detector_name="NicMajorityDownDetector")
        if blind is not None:
            return blind

        faults = check_majority_nic_down(ctx.metric_store.time_series_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no majority NIC down",
            trigger=TriggerType.HARDWARE,
        )
