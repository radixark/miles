"""Tail statistics of the off-policy importance weight w = pi_trainer / mu_rollout, per train step.

The loss metrics are sums reduced across micro-batches and ranks, which cannot carry a max or a quantile,
so the policy loss feeds this collector instead and the train step reduces it once. The split follows
the sampling version (``version_split``) under fully async rollout: the stale tail adds policy drift to the
kernel and routing mismatch that the fresh tail measures.

Reported under ``is/``: count, max, p99, p99.9 (from a log-spaced histogram, 0.5% relative resolution,
clamped to the observed range), the fraction the applied correction changed (``clipfrac``), and the
token-level effective sample size ratio (sum w)^2 / (n * sum w^2) of the raw and the applied weights.
"""

import math

import torch
import torch.distributed as dist

from miles.platforms import current_platform

SPLITS = ("all", "fresh", "stale")
QUANTILES = {"p99": 0.99, "p99_9": 0.999}

_LOG_RANGE = 20.0
_BIN_WIDTH = 0.005
_NUM_BINS = int(2 * _LOG_RANGE / _BIN_WIDTH)
# per split: count, sum w, sum w^2, sum applied, sum applied^2, changed by the correction
_NUM_SUMS = 6


class IsWeightStats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._sums: torch.Tensor | None = None
        self._hist: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    def add(
        self,
        *,
        raw: torch.Tensor,
        active: torch.Tensor,
        applied: torch.Tensor | None = None,
        changed: torch.Tensor | None = None,
        fresh: torch.Tensor | None = None,
        stale: torch.Tensor | None = None,
    ) -> None:
        """``raw``, ``applied`` (the weight the loss used, default ``raw``), ``changed`` (tokens the correction
        altered, default ``applied != raw``) and the masks share one token layout."""
        if torch._C._current_graph_task_id() != -1:
            # A checkpointed loss (--recompute-loss-function) runs again inside backward; count each microbatch once.
            return
        raw = raw.detach().float()
        applied = raw if applied is None else applied.detach().float()
        if self._sums is None:
            device = raw.device
            self._sums = torch.zeros(len(SPLITS), _NUM_SUMS, dtype=torch.float64, device=device)
            self._hist = torch.zeros(len(SPLITS), _NUM_BINS, dtype=torch.float64, device=device)
            self._max = torch.zeros(len(SPLITS), dtype=torch.float64, device=device)
        finite = torch.isfinite(raw)
        log_w = torch.log(raw.clamp_min(torch.finfo(torch.float32).tiny))
        bins = ((log_w + _LOG_RANGE) / _BIN_WIDTH).floor().clamp(0, _NUM_BINS - 1).long()
        bins = torch.where(finite, bins, torch.full_like(bins, _NUM_BINS - 1))
        changed = (applied != raw).double() if changed is None else changed.detach().double()
        active = active.bool()
        for i, mask in enumerate((active, fresh, stale)):
            if mask is None:
                continue
            mask = mask.bool() & active
            weight = mask.double()
            w = torch.where(mask & finite, raw.double(), raw.new_zeros((), dtype=torch.float64))
            a = torch.where(mask, applied.double(), raw.new_zeros((), dtype=torch.float64))
            self._sums[i] += torch.stack(
                [weight.sum(), w.sum(), (w * w).sum(), a.sum(), (a * a).sum(), (changed * weight).sum()]
            )
            # torch.bincount has no deterministic CUDA kernel, so deterministic runs could not use it.
            uniq, counts = torch.unique(bins[mask], return_counts=True)
            self._hist[i, uniq] += counts.to(self._hist.dtype)
            masked_raw = torch.where(mask & finite, raw.double(), raw.new_zeros((), dtype=torch.float64))
            if masked_raw.numel():
                self._max[i] = torch.maximum(self._max[i], masked_raw.max())

    def reduce_and_reset(self, groups_inner_to_outer, all_reduce) -> dict[str, float]:
        """Reduce over the data/context-parallel groups and return the step's metrics.

        Collective: every rank of the last pipeline stage must call it, including ranks that saw no tokens.
        ``all_reduce(tensor, groups, op)`` is ``MultiPGUtil.all_reduce``.
        """
        if self._sums is None:
            # A rank that saw no tokens; CPU-only unit tests have no accelerator.
            device = current_platform.current_device() if torch.accelerator.is_available() else None
            self._sums = torch.zeros(len(SPLITS), _NUM_SUMS, dtype=torch.float64, device=device)
            self._hist = torch.zeros(len(SPLITS), _NUM_BINS, dtype=torch.float64, device=device)
            self._max = torch.zeros(len(SPLITS), dtype=torch.float64, device=device)
        packed = torch.cat([self._sums.flatten(), self._hist.flatten()])
        all_reduce(packed, groups_inner_to_outer, dist.ReduceOp.SUM)
        all_reduce(self._max, groups_inner_to_outer, dist.ReduceOp.MAX)
        sums = packed[: self._sums.numel()].view_as(self._sums).tolist()
        hist = packed[self._sums.numel() :].view_as(self._hist)
        maxes = self._max.tolist()
        self.reset()
        return summarize(sums, hist, maxes)


def summarize(sums: list[list[float]], hist: torch.Tensor, maxes: list[float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for i, split in enumerate(SPLITS):
        count, sum_w, sum_w2, sum_a, sum_a2, changed = sums[i]
        metrics[f"is/{split}_count"] = count
        if count == 0:
            continue
        metrics[f"is/{split}_max"] = maxes[i]
        cumulative = hist[i].cumsum(0)
        min_w = math.exp(int((cumulative > 0).nonzero()[0]) * _BIN_WIDTH - _LOG_RANGE)
        for name, q in QUANTILES.items():
            index = int(
                torch.searchsorted(
                    cumulative, torch.tensor(q * count, dtype=cumulative.dtype, device=cumulative.device)
                )
            )
            upper = math.exp((min(index, _NUM_BINS - 1) + 1) * _BIN_WIDTH - _LOG_RANGE)
            metrics[f"is/{split}_{name}"] = min(max(upper, min_w), maxes[i])
        metrics[f"is/{split}_clipfrac"] = changed / count
        metrics[f"is/{split}_ess_ratio_raw"] = sum_w**2 / (count * sum_w2) if sum_w2 > 0 else 0.0
        metrics[f"is/{split}_ess_ratio"] = sum_a**2 / (count * sum_a2) if sum_a2 > 0 else 0.0
    return metrics


IS_WEIGHT_STATS = IsWeightStats()
