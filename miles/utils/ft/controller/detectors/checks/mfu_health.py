from dataclasses import dataclass

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol


@dataclass(frozen=True)
class MfuHealthStatus:
    avg_mfu: float
    baseline: float
    threshold: float
    is_declining: bool


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
    """
    recent = mini_wandb.query_last_n_steps("mfu", last_n=consecutive_steps)
    if len(recent) < consecutive_steps:
        return None

    avg_mfu = sum(sv.value for sv in recent) / len(recent)

    computed_baseline = _compute_baseline(
        mini_wandb,
        explicit_baseline=baseline,
        baseline_steps=baseline_steps,
        consecutive_steps=consecutive_steps,
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
) -> float:
    if explicit_baseline is not None:
        return explicit_baseline

    total_needed = baseline_steps + consecutive_steps
    all_data = mini_wandb.query_last_n_steps("mfu", last_n=total_needed)

    baseline_data = all_data[:-consecutive_steps]
    if not baseline_data:
        return 0.0

    return sum(sv.value for sv in baseline_data) / len(baseline_data)
