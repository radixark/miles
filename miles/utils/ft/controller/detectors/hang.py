from datetime import timedelta

from pydantic import ConfigDict, field_validator

from miles.utils.ft.models.metric_names import (
    AGENT_HEARTBEAT,
    PHASE_CHECKPOINT_SAVING,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.protocols.platform import JobStatus


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


class HangDetector(BaseFaultDetector):
    def __init__(self, config: HangDetectorConfig | None = None) -> None:
        self._config = config or HangDetectorConfig()

    def evaluate(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision.no_fault(reason="job not running, skipping hang check")

        is_checkpoint_saving = self._is_checkpoint_saving(ctx.metric_store)
        timeout_minutes = (
            self._config.checkpoint_saving_timeout_minutes
            if is_checkpoint_saving
            else self._config.training_timeout_minutes
        )

        heartbeat_changes = self._get_heartbeat_changes(ctx.metric_store, window_minutes=timeout_minutes)
        if heartbeat_changes is None:
            return Decision.no_fault(reason="no heartbeat data available")

        if heartbeat_changes == 0:
            phase_info = "checkpoint_saving" if is_checkpoint_saving else "training"
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"heartbeat stalled for {timeout_minutes}min during {phase_info}",
                trigger=TriggerType.HANG,
            )

        return Decision.no_fault(reason="heartbeat progressing normally")

    def _is_checkpoint_saving(self, metric_store: MetricQueryProtocol) -> bool:
        df = metric_store.query_latest(TRAINING_PHASE, label_filters={"rank": "0"})
        if df is None or df.is_empty():
            return False

        for row in df.iter_rows(named=True):
            if row["value"] == PHASE_CHECKPOINT_SAVING:
                return True

        return False

    def _get_heartbeat_changes(
        self, metric_store: MetricQueryProtocol, window_minutes: int,
    ) -> float | None:
        df = metric_store.changes(
            AGENT_HEARTBEAT,
            window=timedelta(minutes=window_minutes),
            label_filters={"rank": "0"},
        )
        if df is None or df.is_empty():
            return None

        return df["value"][0]
