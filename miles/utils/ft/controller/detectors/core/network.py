from datetime import timedelta

from pydantic import ConfigDict, field_validator

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import check_nic_down_in_window
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import Decision, TriggerType


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
    is_critical = True

    def __init__(self, config: NetworkAlertDetectorConfig | None = None) -> None:
        self._config = config or NetworkAlertDetectorConfig()
        self._alert_window = timedelta(minutes=self._config.alert_window_minutes)
        self._alert_threshold = self._config.alert_threshold

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = check_nic_down_in_window(
            ctx.metric_store,
            window=self._alert_window,
            threshold=self._alert_threshold,
        )
        return Decision.from_node_faults(
            faults,
            fallback_reason="NIC alerts below threshold",
            trigger=TriggerType.NETWORK,
        )
