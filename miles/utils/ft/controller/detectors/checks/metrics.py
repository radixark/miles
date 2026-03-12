"""Training metric checks (loss sanity, etc.)."""

from __future__ import annotations

import math

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol


def get_non_finite_loss(mini_wandb: TrainingMetricStoreProtocol) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss")
    if latest_loss is not None and not math.isfinite(latest_loss):
        return latest_loss
    return None
