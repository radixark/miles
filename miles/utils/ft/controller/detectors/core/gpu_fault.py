from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.gpu.checks import check_gpu_faults
from miles.utils.ft.controller.types import Decision, TriggerType


class GpuFaultDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        faults = check_gpu_faults(ctx.metric_store.time_series_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no GPU faults",
            trigger=TriggerType.HARDWARE,
        )
