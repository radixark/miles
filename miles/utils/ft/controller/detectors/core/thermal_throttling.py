from dataclasses import dataclass, field

import polars as pl
from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.mfu_health import check_mfu_health
from miles.utils.ft.utils.metric_names import DCGM_FI_DEV_GPU_TEMP
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


@dataclass
class _GpuOutlier:
    node_id: str
    gpu: str
    temperature: float


@dataclass
class _TemperatureOutlierResult:
    node_ids: list[str]
    gpu_outliers: list[_GpuOutlier] = field(default_factory=list)


class ThermalThrottlingDetector(BaseFaultDetector):
    def __init__(self, config: ThermalThrottlingDetectorConfig | None = None) -> None:
        self._config = config or ThermalThrottlingDetectorConfig()

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        result = _find_temperature_outlier_gpus(
            metric_store=ctx.metric_store.time_series_store,
            active_node_ids=ctx.active_node_ids,
            delta_threshold=self._config.temperature_delta_threshold,
        )
        if not result.node_ids:
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
                reason=f"temperature outlier on {result.node_ids} but MFU is healthy",
            )

        gpu_detail = ", ".join(
            f"{o.node_id}:gpu{o.gpu}={o.temperature:.0f}°C"
            for o in result.gpu_outliers
        )
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=result.node_ids,
            reason=(
                f"thermal throttling [{gpu_detail}]: "
                f"MFU decline ({mfu.avg_mfu:.4f} < {mfu.threshold:.4f})"
            ),
            trigger=TriggerType.HARDWARE,
        )


def _find_temperature_outlier_gpus(
    metric_store: TimeSeriesQueryProtocol,
    active_node_ids: frozenset[str],
    delta_threshold: float,
) -> _TemperatureOutlierResult:
    """Detect individual GPUs whose temperature exceeds the cluster average
    by more than *delta_threshold*.

    Per-GPU detection (paper §8.1.1) ensures a single overheating GPU is
    caught even when other GPUs on the same node are normal.  Returns the
    affected node IDs (for recovery) plus per-GPU detail (for logging).
    """
    empty = _TemperatureOutlierResult(node_ids=[])
    if not active_node_ids:
        return empty

    df = metric_store.query_latest(DCGM_FI_DEV_GPU_TEMP)
    if df is None or df.is_empty():
        return empty

    df = df.filter(pl.col("node_id").is_in(active_node_ids))
    if df.is_empty():
        return empty

    overall_avg = df["value"].mean()
    threshold = overall_avg + delta_threshold

    hot_gpus = df.filter(pl.col("value") > threshold).sort("value", descending=True)
    if hot_gpus.is_empty():
        return empty

    gpu_outliers = [
        _GpuOutlier(
            node_id=row["node_id"],
            gpu=row["gpu"],
            temperature=row["value"],
        )
        for row in hot_gpus.iter_rows(named=True)
    ]

    seen: set[str] = set()
    node_ids: list[str] = []
    for outlier in gpu_outliers:
        if outlier.node_id not in seen:
            seen.add(outlier.node_id)
            node_ids.append(outlier.node_id)

    return _TemperatureOutlierResult(node_ids=node_ids, gpu_outliers=gpu_outliers)
