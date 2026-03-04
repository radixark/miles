from miles.utils.ft.controller.detectors._metric_names import (
    TRAINING_ITERATION,
    TRAINING_JOB_STATUS,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_JOB_STATUS_RUNNING: int = 1
_PHASE_CHECKPOINT_SAVING: float = 2.0


class HangDetector(BaseFaultDetector):
    def __init__(
        self,
        training_timeout_minutes: int = 10,
        checkpoint_saving_timeout_minutes: int = 30,
    ) -> None:
        self._training_timeout_minutes = training_timeout_minutes
        self._checkpoint_saving_timeout_minutes = checkpoint_saving_timeout_minutes

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        if not self._is_job_running(metric_store):
            return Decision(action=ActionType.NONE, reason="job not running, skipping hang check")

        is_checkpoint_saving = self._is_checkpoint_saving(metric_store)
        timeout_minutes = (
            self._checkpoint_saving_timeout_minutes
            if is_checkpoint_saving
            else self._training_timeout_minutes
        )

        changes = self._get_iteration_changes(metric_store, window_minutes=timeout_minutes)
        if changes is None:
            return Decision(action=ActionType.NONE, reason="no iteration data available")

        if changes == 0:
            phase_info = "checkpoint_saving" if is_checkpoint_saving else "training"
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"iteration stalled for {timeout_minutes}min during {phase_info}",
                trigger="hang",
            )

        return Decision(action=ActionType.NONE, reason="iteration progressing normally")

    def _is_job_running(self, metric_store: MetricStoreProtocol) -> bool:
        df = metric_store.instant_query(f"{TRAINING_JOB_STATUS} == {_JOB_STATUS_RUNNING}")
        return not df.is_empty()

    def _is_checkpoint_saving(self, metric_store: MetricStoreProtocol) -> bool:
        df = metric_store.instant_query(TRAINING_PHASE)
        if df.is_empty():
            return False

        for row in df.iter_rows(named=True):
            if row.get("rank") == "0" and row["value"] == _PHASE_CHECKPOINT_SAVING:
                return True

        return False

    def _get_iteration_changes(
        self, metric_store: MetricStoreProtocol, window_minutes: int,
    ) -> float | None:
        query = f'changes({TRAINING_ITERATION}{{rank="0"}}[{window_minutes}m])'
        df = metric_store.instant_query(query)
        if df.is_empty():
            return None

        return df["value"][0]
