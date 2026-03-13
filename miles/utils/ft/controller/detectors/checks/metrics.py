"""Training metric checks (loss sanity, etc.)."""

from __future__ import annotations

import logging
import math

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol

logger = logging.getLogger(__name__)


def get_non_finite_loss(mini_wandb: TrainingMetricStoreProtocol) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss")
    if latest_loss is not None and not math.isfinite(latest_loss):
        logger.info("detector_check: non-finite loss detected: %s", latest_loss)
        return latest_loss
    return None
