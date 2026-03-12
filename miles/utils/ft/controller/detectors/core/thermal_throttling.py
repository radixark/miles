import polars as pl
from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.mfu_health import check_mfu_health
from miles.utils.ft.controller.metrics.metric_names import DCGM_FI_DEV_GPU_TEMP
from miles.utils.ft.controller.types import ActionType, Decision, TimeSeriesQueryProtocol, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel


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

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        hot_node_ids = _find_temperature_outlier_nodes(
            metric_store=ctx.metric_store.time_series_store,
            active_node_ids=ctx.active_node_ids,
            delta_threshold=self._config.temperature_delta_threshold,
        )
        if not hot_node_ids:
            return Decision.no_fault(reason="no temperature outlier")

        cfg = self._config
        mfu = check_mfu_health(
            ctx.metric_store.mini_wandb,
            consecutive_steps=cfg.mfu_consecutive_steps,
            threshold_ratio=cfg.mfu_decline_threshold_ratio,
            baseline=cfg.mfu_baseline,
            baseline_steps=cfg.mfu_baseline_steps,
        )
        if mfu is None or not mfu.is_declining:
            return Decision.no_fault(
                reason=f"temperature outlier on {hot_node_ids} but MFU is healthy",
            )

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=hot_node_ids,
            reason=(
                f"thermal throttling on {hot_node_ids}: " f"MFU decline ({mfu.avg_mfu:.4f} < {mfu.threshold:.4f})"
            ),
            trigger=TriggerType.HARDWARE,
        )


def _find_temperature_outlier_nodes(
    metric_store: TimeSeriesQueryProtocol,
    active_node_ids: set[str],
    delta_threshold: float,
) -> list[str]:
    """Return all nodes whose hottest GPU exceeds the cluster-wide average
    by more than *delta_threshold*.

    Uses per-node **max** GPU temperature (not mean) so that a single
    overheating GPU is never masked by cooler siblings on the same node.
    The cluster baseline is the mean of all individual GPU samples to
    avoid being skewed by outlier nodes.
    """
    if not active_node_ids:
        return []

    df = metric_store.query_latest(DCGM_FI_DEV_GPU_TEMP)
    if df is None or df.is_empty():
        return []

    df = df.filter(pl.col("node_id").is_in(active_node_ids))
    if df.is_empty():
        return []

    overall_avg = df["value"].mean()

    node_max = df.group_by("node_id").agg(pl.col("value").max().alias("max_temp"))
    outliers = node_max.filter(pl.col("max_temp") > overall_avg + delta_threshold)
    if outliers.is_empty():
        return []

    return outliers.sort("max_temp", descending=True)["node_id"].to_list()
