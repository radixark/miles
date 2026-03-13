"""Training metric checks (loss sanity, etc.)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol


def get_non_finite_loss(mini_wandb: TrainingMetricStoreProtocol) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss")
    if latest_loss is not None and not math.isfinite(latest_loss):
        return latest_loss
    return None


@dataclass(frozen=True)
class SpikeResult:
    metric_name: str
    current_avg: float
    baseline_avg: float
    ratio: float


def check_metric_spike(
    mini_wandb: TrainingMetricStoreProtocol,
    *,
    metric_name: str,
    recent_steps: int,
    baseline_steps: int,
    spike_threshold: float,
) -> SpikeResult | None:
    """Detect if *metric_name* has spiked relative to its historical baseline.

    Compares the average of the last *recent_steps* to the average of
    the preceding *baseline_steps*.  Returns a SpikeResult if the ratio
    exceeds *spike_threshold*, None otherwise.

    Non-finite values (NaN/Inf) in either window cause the check to
    return None (those are handled by NanLossDetector separately).
    """
    total_needed = baseline_steps + recent_steps
    all_data = mini_wandb.query_last_n_steps(metric_name, last_n=total_needed)
    if len(all_data) < total_needed:
        return None

    baseline_data = all_data[:baseline_steps]
    recent_data = all_data[baseline_steps:]

    baseline_values = [sv.value for sv in baseline_data]
    recent_values = [sv.value for sv in recent_data]

    if not all(math.isfinite(v) for v in baseline_values + recent_values):
        return None

    baseline_avg = sum(baseline_values) / len(baseline_values)
    if baseline_avg <= 0:
        return None

    current_avg = sum(recent_values) / len(recent_values)
    ratio = current_avg / baseline_avg

    if ratio >= spike_threshold:
        return SpikeResult(
            metric_name=metric_name,
            current_avg=current_avg,
            baseline_avg=baseline_avg,
            ratio=ratio,
        )
    return None
