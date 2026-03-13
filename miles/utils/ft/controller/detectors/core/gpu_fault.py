from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, check_metric_blind
from miles.utils.ft.controller.detectors.checks.gpu.checks import check_gpu_faults
from miles.utils.ft.controller.types import Decision, TriggerType
from miles.utils.ft.utils.metric_names import GPU_AVAILABLE


class GpuFaultDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        blind = check_metric_blind(ctx, GPU_AVAILABLE, detector_name="GpuFaultDetector")
        if blind is not None:
            return blind

        faults = check_gpu_faults(ctx.metric_store.time_series_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no GPU faults",
            trigger=TriggerType.HARDWARE,
        )
