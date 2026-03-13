import logging
from datetime import timedelta

from pydantic import ConfigDict, field_validator

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.training_metric_filters import build_training_metric_filters
from miles.utils.ft.controller.metrics.mini_prometheus.query import AmbiguousSeriesError
from miles.utils.ft.controller.types import ActionType, Decision, TimeSeriesQueryProtocol, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.metric_names import AGENT_HEARTBEAT, PHASE_CHECKPOINT_SAVING, PHASE_TRAINING, TRAINING_PHASE

logger = logging.getLogger(__name__)


class HangDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_timeout_minutes: int = 10
    checkpoint_saving_timeout_minutes: int = 30
    startup_timeout_minutes: int = 15

    @field_validator("training_timeout_minutes", "checkpoint_saving_timeout_minutes", "startup_timeout_minutes")
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
            logger.debug("hang_detector: job not running (status=%s), skipping", ctx.job_status)
            return Decision.no_fault(reason="job not running, skipping hang check")

        if ctx.active_run_id is None:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="active_run_id not established, run-scoped metric queries unreliable",
                trigger=TriggerType.TELEMETRY_BLIND,
                notify_deduplicator_id="hang:active_run_id_missing",
            )

        label_filters = build_training_metric_filters(rank="0", run_id=ctx.active_run_id)
        within_startup = ctx.seconds_since_run_start < self._config.startup_timeout_minutes * 60

        phase = self._get_current_phase(ctx.metric_store.time_series_store, label_filters=label_filters)
        if phase is None:
            return Decision.no_fault(reason="phase unknown, skipping hang check")
        timeout_attr = _PHASE_TIMEOUT_ATTR.get(phase)
        if timeout_attr is None:
            return Decision.no_fault(reason=f"unknown training phase {phase}, skipping hang check")
        timeout_minutes: int = getattr(self._config, timeout_attr)

        has_any_data, heartbeat_changes = self._get_heartbeat_changes(
            ctx.metric_store.time_series_store,
            window_minutes=timeout_minutes,
            label_filters=label_filters,
        )

        if not has_any_data:
            if within_startup:
                logger.debug("hang_detector: no heartbeat data but within startup grace, ignoring")
                return Decision.no_fault(reason="within startup grace period, ignoring missing heartbeat")
            logger.warning("hang_detector: no heartbeat data from rank-0 after grace period")
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="no heartbeat data from rank-0 agent after grace period",
                trigger=TriggerType.TELEMETRY_BLIND,
                notify_deduplicator_id=f"hang:no_heartbeat:run={ctx.active_run_id}",
            )

        if heartbeat_changes is None:
            return Decision.no_fault(reason="heartbeat data insufficient for hang judgment")

        if heartbeat_changes == 0:
            if within_startup:
                logger.debug("hang_detector: heartbeat stalled but within startup grace, ignoring")
                return Decision.no_fault(reason="within startup grace period, ignoring stalled heartbeat")
            phase_label = _PHASE_LABEL[phase]
            logger.info(
                "hang_detector: heartbeat stalled for %dmin during %s, entering recovery",
                timeout_minutes,
                phase_label,
            )
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"heartbeat stalled for {timeout_minutes}min during {phase_label}",
                trigger=TriggerType.HANG,
            )

        return Decision.no_fault(reason="heartbeat progressing normally")

    def _get_current_phase(
        self,
        metric_store: TimeSeriesQueryProtocol,
        *,
        label_filters: dict[str, str],
    ) -> float | None:
        try:
            df = metric_store.query_single_latest(TRAINING_PHASE, label_filters=label_filters)
        except AmbiguousSeriesError:
            logger.warning(
                "ambiguous_training_phase_series: phase unknown, skipping hang check",
                exc_info=True,
            )
            return None

        if df.is_empty():
            return None

        return df.row(0, named=True)["value"]

    def _get_heartbeat_changes(
        self,
        metric_store: TimeSeriesQueryProtocol,
        window_minutes: int,
        *,
        label_filters: dict[str, str],
    ) -> tuple[bool, float | None]:
        window = timedelta(minutes=window_minutes)

        df = metric_store.changes(
            AGENT_HEARTBEAT,
            window=window,
            label_filters=label_filters,
        )
        if df is None or df.is_empty():
            return (False, None)

        min_changes = df["value"].min()
        if min_changes == 0:
            count_df = metric_store.count_over_time(
                AGENT_HEARTBEAT,
                window=window,
                label_filters=label_filters,
            )
            if count_df is not None and not count_df.is_empty():
                min_count = count_df["value"].min()
                if min_count < 2:
                    return (True, None)

        return (True, min_changes)
