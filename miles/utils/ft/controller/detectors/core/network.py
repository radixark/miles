from datetime import timedelta

from pydantic import ConfigDict, field_validator

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, check_metric_blind
from miles.utils.ft.controller.detectors.checks.hardware import check_nic_down_in_window, check_nic_persistent_down
from miles.utils.ft.controller.types import Decision, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.metric_names import NODE_NETWORK_UP


class NetworkAlertDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_window_minutes: float = 5.0
    alert_threshold: int = 2

    @field_validator("alert_window_minutes")
    @classmethod
    def _window_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("alert_window_minutes must be positive")
        return value

    @field_validator("alert_threshold")
    @classmethod
    def _threshold_must_be_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("alert_threshold must be >= 1")
        return value


class NetworkAlertDetector(BaseFaultDetector):
    def __init__(self, config: NetworkAlertDetectorConfig | None = None) -> None:
        self._config = config or NetworkAlertDetectorConfig()
        self._alert_window = timedelta(minutes=self._config.alert_window_minutes)
        self._alert_threshold = self._config.alert_threshold

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        blind = check_metric_blind(ctx, NODE_NETWORK_UP, detector_name="NetworkAlertDetector")
        if blind is not None:
            return blind

        flap_faults = check_nic_down_in_window(
            ctx.metric_store.time_series_store,
            window=self._alert_window,
            threshold=self._alert_threshold,
        )
        persistent_faults = check_nic_persistent_down(
            ctx.metric_store.time_series_store,
            window=self._alert_window,
        )

        seen_nodes: set[str] = set()
        merged: list = []
        for fault in flap_faults + persistent_faults:
            if fault.node_id not in seen_nodes:
                seen_nodes.add(fault.node_id)
                merged.append(fault)

        return Decision.from_node_faults(
            merged,
            fallback_reason="NIC alerts below threshold",
            trigger=TriggerType.NETWORK,
        )
