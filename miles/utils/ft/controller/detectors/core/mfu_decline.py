from datetime import datetime, timedelta, timezone

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.mfu_health import check_mfu_health
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.metrics import TrainingMetricStoreProtocol


class MfuDeclineDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mfu_baseline: float | None = None
    mfu_threshold_ratio: float = Field(default=0.8, gt=0, le=1)
    consecutive_steps: int = Field(default=10, ge=1)
    decline_timeout_minutes: float = Field(default=30.0, gt=0)
    baseline_steps: int = Field(default=50, ge=1)
    mfu_absolute_minimum: float = Field(default=0.0, ge=0)


class MfuDeclineDetector(BaseFaultDetector):
    def __init__(self, config: MfuDeclineDetectorConfig | None = None) -> None:
        self._config = config or MfuDeclineDetectorConfig()

    def evaluate(self, ctx: DetectorContext) -> Decision:
        cfg = self._config

        mfu = check_mfu_health(
            ctx.mini_wandb,
            consecutive_steps=cfg.consecutive_steps,
            threshold_ratio=cfg.mfu_threshold_ratio,
            baseline=cfg.mfu_baseline,
            baseline_steps=cfg.baseline_steps,
        )
        if mfu is None:
            return Decision.no_fault(reason="insufficient MFU data or no baseline")

        if cfg.mfu_absolute_minimum > 0 and mfu.avg_mfu < cfg.mfu_absolute_minimum:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU {mfu.avg_mfu:.4f} below absolute minimum {cfg.mfu_absolute_minimum:.4f}",
                trigger=TriggerType.MISC,
            )

        if not mfu.is_declining:
            return Decision.no_fault(reason="MFU within acceptable range")

        # ByteRobust (arxiv 2509.16293 Sec.5) uses stack trace aggregation to
        # localize faulty machines on MFU decline; for simplicity we notify human.
        decline_summary = f"{mfu.avg_mfu:.4f} < {mfu.threshold:.4f}"

        elapsed_minutes = self._compute_decline_duration_minutes(ctx.mini_wandb, mfu.threshold)
        if elapsed_minutes >= cfg.decline_timeout_minutes:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU decline ({decline_summary}) persisted for {elapsed_minutes:.1f}min without identifiable cause",
                trigger=TriggerType.MISC,
            )

        return Decision.no_fault(
            reason=f"MFU declining ({decline_summary}), monitoring ({elapsed_minutes:.1f}min)",
        )

    def _compute_decline_duration_minutes(
        self, mini_wandb: TrainingMetricStoreProtocol, threshold: float,
    ) -> float:
        """Derive how long MFU has been below *threshold* from time-series data.

        Queries a window wider than the timeout so the "last healthy" reading
        is visible even when the decline started exactly at the timeout boundary.
        """
        lookup_window = timedelta(minutes=self._config.decline_timeout_minutes * 2)
        timed_mfu = mini_wandb.query_time_window("mfu", window=lookup_window)
        if not timed_mfu:
            return 0.0

        healthy_times = [ts for _, ts, value in timed_mfu if value >= threshold]
        decline_start = healthy_times[-1] if healthy_times else timed_mfu[0].timestamp
        return (datetime.now(timezone.utc) - decline_start).total_seconds() / 60
