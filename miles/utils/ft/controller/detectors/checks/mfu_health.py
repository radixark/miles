import math
from dataclasses import dataclass

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol


@dataclass(frozen=True)
class MfuHealthStatus:
    avg_mfu: float
    baseline: float
    threshold: float
    is_declining: bool
    telemetry_valid: bool = True


def check_mfu_health(
    mini_wandb: TrainingMetricStoreProtocol,
    *,
    consecutive_steps: int,
    threshold_ratio: float,
    baseline: float | None,
    baseline_steps: int,
) -> MfuHealthStatus | None:
    """Check whether MFU is declining relative to a baseline.

    Returns None when there is insufficient data or no valid baseline.
    Returns MfuHealthStatus with telemetry_valid=False when non-finite
    values (NaN/Inf) are present in either the recent or baseline window.
    """
    recent = mini_wandb.query_last_n_steps("mfu", last_n=consecutive_steps)
    if len(recent) < consecutive_steps:
        return None

    recent_finite = [sv for sv in recent if math.isfinite(sv.value)]
    if len(recent_finite) < len(recent):
        return MfuHealthStatus(
            avg_mfu=float("nan"),
            baseline=0.0,
            threshold=0.0,
            is_declining=False,
            telemetry_valid=False,
        )

    avg_mfu = sum(sv.value for sv in recent_finite) / len(recent_finite)

    computed_baseline = _compute_baseline(
        mini_wandb,
        explicit_baseline=baseline,
        baseline_steps=baseline_steps,
        consecutive_steps=consecutive_steps,
    )
    if computed_baseline is None:
        return MfuHealthStatus(
            avg_mfu=avg_mfu,
            baseline=0.0,
            threshold=0.0,
            is_declining=False,
            telemetry_valid=False,
        )
    if computed_baseline <= 0:
        return None

    thresh = computed_baseline * threshold_ratio
    return MfuHealthStatus(
        avg_mfu=avg_mfu,
        baseline=computed_baseline,
        threshold=thresh,
        is_declining=avg_mfu < thresh,
    )


def _compute_baseline(
    mini_wandb: TrainingMetricStoreProtocol,
    *,
    explicit_baseline: float | None,
    baseline_steps: int,
    consecutive_steps: int,
) -> float | None:
    """Compute MFU baseline. Returns None if non-finite values are present."""
    if explicit_baseline is not None:
        if not math.isfinite(explicit_baseline):
            return None
        return explicit_baseline

    total_needed = baseline_steps + consecutive_steps
    all_data = mini_wandb.query_last_n_steps("mfu", last_n=total_needed)

    baseline_data = all_data[:-consecutive_steps]
    if not baseline_data:
        return 0.0

    finite_values = [sv for sv in baseline_data if math.isfinite(sv.value)]
    if len(finite_values) < len(baseline_data):
        return None

    return sum(sv.value for sv in finite_values) / len(finite_values)
