import logging
from typing import Optional, Callable
import torch.distributed as dist

from miles.utils import tracking_utils
import torch

logger = logging.getLogger(__name__)

# list of metrics that need to extend
_EXTEND_METRICS = ["advantages", "pg_loss"]


def update_extend_list(new_metrics: list):
    global _EXTEND_METRICS
    _EXTEND_METRICS = new_metrics


def process_metric(log_dict, key: str = "", metric: torch.Tensor = None, group=None, amount=-1):
    """Processes and records metrics into a target dictionary.

    This function supports two primary modes of operation:
    1. Simple Mean: Pass the summed results in 'metric' and specify the divisor in 'amount'.
    2. Extended Metrics/Reduction: Pass a vector of raw sample means in 'metric' to
       perform a reduction operation across a group.

    Args:
        log_dict (dict): The destination dictionary where the processed metrics will be stored.
        key (str): The name/identifier for the metric.
        metric (torch.Tensor, optional): A 1D tensor containing the metric data.
            Can be a single summed value or a vector of sample means. Defaults to None.
        group (optional): The process group or category used for reduction operations.
            Defaults to None.
        amount (int, optional): The divisor used for calculating the average.
            If -1, indicates a custom or pre-calculated reduction logic. Defaults to -1.

    Returns:
        None: The results are updated directly in the log_dict.
    """
    if not any(item in key for item in _EXTEND_METRICS):
        log_dict[key] = _mean_reduce(metric, group, amount)
    else:
        mean, std_, max_, min_ = _extend_reduce(metric, group)
        log_dict[key] = mean
        log_dict[f"{key}/max"] = max_
        log_dict[f"{key}/min"] = min_
        log_dict[f"{key}/std"] = std_


def _mean_reduce(local_vals: torch.Tensor, group, amount):
    if local_vals.numel() == 0:
        # default value for empty impl
        local_sum = torch.tensor(0.0, device=local_vals.device)
        local_count = torch.tensor(0.0, device=local_vals.device)
    else:
        local_sum = local_vals.sum()
        local_count = torch.tensor(local_vals.numel(), device=local_vals.device, dtype=local_vals.dtype)

    metric = torch.stack([local_sum, local_count]).to(local_vals.device)
    dist.all_reduce(metric, op=dist.ReduceOp.SUM, group=group)
    global_sum, global_count = metric.tolist()
    if amount > 0:
        global_count = amount
    return global_sum / global_count


def _extend_reduce(local_vals: torch.Tensor, group):
    if local_vals.numel() > 0:
        max_ = local_vals.max()
        min_ = local_vals.min()
        local_sum = local_vals.sum()
        local_sum_sq = (local_vals ** 2).sum()
    else:
        # default value for empty impl
        max_ = torch.tensor(float("-inf"), device=local_vals.device, dtype=local_vals.dtype)
        min_ = torch.tensor(float("inf"), device=local_vals.device, dtype=local_vals.dtype)
        local_sum = torch.tensor(0.0, device=local_vals.device, dtype=local_vals.dtype)
        local_sum_sq = torch.tensor(0.0, device=local_vals.device, dtype=local_vals.dtype)
    local_count = torch.tensor(local_vals.numel(), dtype=torch.float32, device=local_vals.device)

    metric = torch.stack([local_sum, local_sum_sq, local_count])

    dist.all_reduce(metric, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(min_, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(max_, op=dist.ReduceOp.MAX, group=group)

    global_sum, global_sum_sq, n = metric.tolist()
    if n == 0:
        return 0.0, 0.0, max_, min_

    global_mean = global_sum / n
    global_var = (global_sum_sq / n) - (global_mean ** 2)
    global_std = (global_var ** 0.5) if global_var > 0 else 0.0
    return global_mean, global_std, max_, min_
