from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from miles.utils.ft.models import Decision
from miles.utils.ft.protocols.metrics import MetricQueryProtocol, TrainingMetricStoreProtocol
from miles.utils.ft.protocols.platform import JobStatus


@dataclass
class DetectorContext:
    metric_store: MetricQueryProtocol
    mini_wandb: TrainingMetricStoreProtocol
    rank_placement: dict[int, str]
    job_status: JobStatus


class BaseFaultDetector(ABC):
    """Base class for fault detectors.

    Detectors must be **stateless**: ``evaluate()`` must derive its answer
    entirely from the data available in ``DetectorContext`` (metric stores,
    job status, rank placement).  No mutable instance state should be
    accumulated across calls.  Constructor parameters (thresholds, config)
    are fine because they are immutable after init.
    """

    is_critical: bool = False

    @abstractmethod
    def evaluate(self, ctx: DetectorContext) -> Decision: ...


def get_non_finite_loss(mini_wandb: TrainingMetricStoreProtocol) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss")
    if latest_loss is not None and not math.isfinite(latest_loss):
        return latest_loss
    return None
