from datetime import timedelta

from pydantic import ConfigDict, field_validator

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.metrics.metric_names import (
    AGENT_HEARTBEAT,
    PHASE_CHECKPOINT_SAVING,
    PHASE_TRAINING,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.types import ActionType, Decision, MetricQueryProtocol, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel


class HangDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_timeout_minutes: int = 10
    checkpoint_saving_timeout_minutes: int = 30

    @field_validator("training_timeout_minutes", "checkpoint_saving_timeout_minutes")
    @classmethod
    def _must_be_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


_PHASE_TIMEOUT_ATTR: dict[float, str] = {
    PHASE_CHECKPOINT_SAVING: "checkpoint_saving_timeout_minutes",
    PHASE_TRAINING: "training_timeout_minutes",
}

_PHASE_LABEL: dict[float, str] = {
    PHASE_CHECKPOINT_SAVING: "checkpoint_saving",
    PHASE_TRAINING: "training",
}


class HangDetector(BaseFaultDetector):
    def __init__(self, config: HangDetectorConfig | None = None) -> None:
        self._config = config or HangDetectorConfig()

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision.no_fault(reason="job not running, skipping hang check")

        phase = self._get_current_phase(ctx.metric_store)
        timeout_attr = _PHASE_TIMEOUT_ATTR.get(phase)
        if timeout_attr is None:
            return Decision.no_fault(reason=f"unknown training phase {phase}, skipping hang check")
        timeout_minutes: int = getattr(self._config, timeout_attr)

        heartbeat_changes = self._get_heartbeat_changes(ctx.metric_store, window_minutes=timeout_minutes)
        if heartbeat_changes is None:
            return Decision.no_fault(reason="no heartbeat data available")

        if heartbeat_changes == 0:
            phase_label = _PHASE_LABEL[phase]
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"heartbeat stalled for {timeout_minutes}min during {phase_label}",
                trigger=TriggerType.HANG,
            )

        return Decision.no_fault(reason="heartbeat progressing normally")

    def _get_current_phase(self, metric_store: MetricQueryProtocol) -> float:
        df = metric_store.query_latest(TRAINING_PHASE, label_filters={"rank": "0"})
        if df is None or df.is_empty():
            return PHASE_TRAINING

        return df.row(0, named=True)["value"]

    def _get_heartbeat_changes(
        self,
        metric_store: MetricQueryProtocol,
        window_minutes: int,
    ) -> float | None:
        df = metric_store.changes(
            AGENT_HEARTBEAT,
            window=timedelta(minutes=window_minutes),
            label_filters={"rank": "0"},
        )
        if df is None or df.is_empty():
            return None

        return df["value"][0]
