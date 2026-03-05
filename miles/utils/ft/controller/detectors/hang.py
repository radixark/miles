from datetime import timedelta

from miles.utils.ft.metric_names import (
    PHASE_CHECKPOINT_SAVING,
    TRAINING_ITERATION,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.protocols.platform import JobStatus


class HangDetector(BaseFaultDetector):
    def __init__(
        self,
        training_timeout_minutes: int = 10,
        checkpoint_saving_timeout_minutes: int = 30,
    ) -> None:
        if training_timeout_minutes < 1:
            raise ValueError(f"training_timeout_minutes must be >= 1, got {training_timeout_minutes}")
        if checkpoint_saving_timeout_minutes < 1:
            raise ValueError(f"checkpoint_saving_timeout_minutes must be >= 1, got {checkpoint_saving_timeout_minutes}")

        self._training_timeout_minutes = training_timeout_minutes
        self._checkpoint_saving_timeout_minutes = checkpoint_saving_timeout_minutes

    def evaluate(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision(action=ActionType.NONE, reason="job not running, skipping hang check")

        is_checkpoint_saving = self._is_checkpoint_saving(ctx.metric_store)
        timeout_minutes = (
            self._checkpoint_saving_timeout_minutes
            if is_checkpoint_saving
            else self._training_timeout_minutes
        )

        iteration_changes = self._get_iteration_changes(ctx.metric_store, window_minutes=timeout_minutes)
        if iteration_changes is None:
            return Decision(action=ActionType.NONE, reason="no iteration data available")

        if iteration_changes == 0:
            phase_info = "checkpoint_saving" if is_checkpoint_saving else "training"
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"iteration stalled for {timeout_minutes}min during {phase_info}",
                trigger=TriggerType.HANG,
            )

        return Decision(action=ActionType.NONE, reason="iteration progressing normally")

    def _is_checkpoint_saving(self, metric_store: MetricQueryProtocol) -> bool:
        df = metric_store.query_latest(TRAINING_PHASE, label_filters={"rank": "0"})
        if df.is_empty():
            return False

        for row in df.iter_rows(named=True):
            if row["value"] == PHASE_CHECKPOINT_SAVING:
                return True

        return False

    def _get_iteration_changes(
        self, metric_store: MetricQueryProtocol, window_minutes: int,
    ) -> float | None:
        df = metric_store.changes(
            TRAINING_ITERATION,
            window=timedelta(minutes=window_minutes),
            label_filters={"rank": "0"},
        )
        if df.is_empty():
            return None

        return df["value"][0]
