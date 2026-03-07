from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.hardware_checks import check_all_hardware_faults
from miles.utils.ft.models.fault import Decision


class HighConfidenceHardwareDetector(BaseFaultDetector):
    is_critical = True

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = check_all_hardware_faults(metric_store=ctx.metric_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no high-confidence hardware faults",
        )
