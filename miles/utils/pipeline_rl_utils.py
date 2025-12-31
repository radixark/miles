from __future__ import annotations

from typing import Any

import torch


def apply_pipeline_rl_lag_mask(
    rollout_data: dict[str, Any],
    *,
    current_version: int,
    max_weight_lag: int,
) -> tuple[int, float]:
    """Compute per-sample weight lag and mask stale samples by zeroing loss masks.

    Uses `weight_version_first` (w_first) if available, otherwise falls back to
    `weight_version_last` (w_last). If a sample's lag is greater than
    `max_weight_lag`, its loss mask is set to all zeros to preserve batch/group
    structure while preventing gradient contribution.

    Returns:
        (exceeds_count, exceeds_frac)
    """

    versions = rollout_data.get("weight_version_first") or rollout_data.get("weight_version_last")
    if not versions:
        return 0, 0.0

    lags: list[int | None] = []
    for v in versions:
        if v is None:
            lags.append(None)
            continue
        try:
            lags.append(int(current_version) - int(v))
        except (TypeError, ValueError):
            lags.append(None)

    rollout_data["weight_lag"] = lags

    if max_weight_lag < 0:
        return 0, 0.0

    exceeds_count = 0
    loss_masks = rollout_data.get("loss_masks")
    if not loss_masks:
        return 0, 0.0

    for i, lag in enumerate(lags):
        if lag is None or lag <= max_weight_lag:
            continue
        exceeds_count += 1
        mask = loss_masks[i]
        if torch.is_tensor(mask):
            mask.zero_()
        else:
            loss_masks[i] = [0] * len(mask)

    exceeds_frac = exceeds_count / max(1, len(lags))
    return exceeds_count, exceeds_frac

