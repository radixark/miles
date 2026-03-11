from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.types import Decision, MetricQueryProtocol, TrainingMetricStoreProtocol

logger = logging.getLogger(__name__)


def _filter_node_ids_by_active(node_ids: list[str], active_node_ids: set[str]) -> list[str]:
    """Keep only node IDs that are in the active training placement."""
    return [n for n in node_ids if n in active_node_ids]


@dataclass
class DetectorContext:
    metric_store: MetricQueryProtocol
    mini_wandb: TrainingMetricStoreProtocol | None = None
    rank_placement: dict[int, str] = field(default_factory=dict)
    job_status: JobStatus | None = None


class BaseFaultDetector(ABC):
    """Base class for fault detectors.

    Detectors must be **stateless**: ``_evaluate_raw()`` must derive its
    answer entirely from the data available in ``DetectorContext`` (metric
    stores, job status, rank placement).  No mutable instance state should
    be accumulated across calls.  Constructor parameters (thresholds,
    config) are fine because they are immutable after init.

    The public ``evaluate()`` method wraps ``_evaluate_raw()`` and filters
    out bad-node IDs that are not in the current training's
    ``rank_placement``, so individual detectors do not need to handle this.
    """

    def evaluate(self, ctx: DetectorContext) -> Decision:
        decision = self._evaluate_raw(ctx)
        if decision.bad_node_ids and ctx.rank_placement:
            active_node_ids = set(ctx.rank_placement.values())
            filtered = _filter_node_ids_by_active(decision.bad_node_ids, active_node_ids)
            if not filtered:
                logger.info(
                    "detector_bad_nodes_not_active detector=%s bad=%s active=%s",
                    type(self).__name__,
                    decision.bad_node_ids,
                    sorted(active_node_ids),
                )
                return Decision.no_fault(
                    reason=f"all bad nodes not active ({type(self).__name__})",
                )
            if len(filtered) != len(decision.bad_node_ids):
                decision = decision.model_copy(update={"bad_node_ids": filtered})
        return decision

    @abstractmethod
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision: ...


def get_non_finite_loss(mini_wandb: TrainingMetricStoreProtocol) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss")
    if latest_loss is not None and not math.isfinite(latest_loss):
        return latest_loss
    return None
