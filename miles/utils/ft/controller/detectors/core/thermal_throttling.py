import polars as pl
from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.mfu_health import check_mfu_health
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.metric_names import DCGM_FI_DEV_GPU_TEMP
from miles.utils.ft.protocols.metrics import MetricQueryProtocol


class ThermalThrottlingDetectorConfig(FtBaseModel):
    """ByteRobust (arxiv 2509.16293 Sec.8.1.1): temperature is the primary
    signal; MFU decline is used to confirm that thermal throttling is actually
    degrading training performance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature_delta_threshold: float = Field(default=20.0, gt=0)
    mfu_decline_threshold_ratio: float = Field(default=0.9, gt=0, le=1)
    mfu_baseline: float | None = None
    mfu_consecutive_steps: int = Field(default=10, ge=1)
    mfu_baseline_steps: int = Field(default=50, ge=1)


class ThermalThrottlingDetector(BaseFaultDetector):
    def __init__(self, config: ThermalThrottlingDetectorConfig | None = None) -> None:
        self._config = config or ThermalThrottlingDetectorConfig()

    def evaluate(self, ctx: DetectorContext) -> Decision:
        hot_node_id = _find_temperature_outlier(
            metric_store=ctx.metric_store,
            rank_placement=ctx.rank_placement,
            delta_threshold=self._config.temperature_delta_threshold,
        )
        if hot_node_id is None:
            return Decision.no_fault(reason="no temperature outlier")

        cfg = self._config
        mfu = check_mfu_health(
            ctx.mini_wandb,
            consecutive_steps=cfg.mfu_consecutive_steps,
            threshold_ratio=cfg.mfu_decline_threshold_ratio,
            baseline=cfg.mfu_baseline,
            baseline_steps=cfg.mfu_baseline_steps,
        )
        if mfu is None or not mfu.is_declining:
            return Decision.no_fault(
                reason=f"temperature outlier on {hot_node_id} but MFU is healthy",
            )

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=[hot_node_id],
            reason=(
                f"thermal throttling on {hot_node_id}: "
                f"MFU decline ({mfu.avg_mfu:.4f} < {mfu.threshold:.4f})"
            ),
            trigger=TriggerType.HARDWARE,
        )


def _find_temperature_outlier(
    metric_store: MetricQueryProtocol,
    rank_placement: dict[int, str],
    delta_threshold: float,
) -> str | None:
    """Return the node whose average GPU temperature exceeds the cluster
    average by more than *delta_threshold*, or None."""
    if not rank_placement:
        return None

    df = metric_store.query_latest(DCGM_FI_DEV_GPU_TEMP)
    if df is None or df.is_empty():
        return None

    node_ids = set(rank_placement.values())
    df = df.filter(pl.col("node_id").is_in(node_ids))
    if df.is_empty():
        return None

    node_avgs = df.group_by("node_id").agg(pl.col("value").mean().alias("avg_temp"))
    overall_avg = node_avgs["avg_temp"].mean()

    outliers = node_avgs.filter(pl.col("avg_temp") > overall_avg + delta_threshold)
    if outliers.is_empty():
        return None

    return outliers.sort("avg_temp", descending=True)["node_id"][0]
