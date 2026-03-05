from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.hardware_checks import (
    CRITICAL_XID_CODES,
    DISK_AVAILABLE_THRESHOLD_BYTES,
    check_all_hardware_faults,
)
from miles.utils.ft.models import Decision


class HighConfidenceHardwareDetector(BaseFaultDetector):
    def __init__(
        self,
        critical_xid_codes: frozenset[int] = CRITICAL_XID_CODES,
        disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
    ) -> None:
        self._critical_xid_codes = critical_xid_codes
        self._disk_available_threshold_bytes = disk_available_threshold_bytes

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = check_all_hardware_faults(
            metric_store=ctx.metric_store,
            critical_xid_codes=self._critical_xid_codes,
            disk_available_threshold_bytes=self._disk_available_threshold_bytes,
        )

        return Decision.from_node_faults(
            faults,
            fallback_reason="no high-confidence hardware faults",
        )
