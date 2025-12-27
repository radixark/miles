import logging
from typing import Optional, Callable
import torch.distributed as dist

from miles.utils import tracking_utils
import torch

logger = logging.getLogger(__name__)

# list of metrics that need to extend
_EXTEND_METRICS = ["advantages", "pg_loss"]


def update_extend_list(new_metrics: list):
    """在程序启动或初始化 Trainer 时调用，更新感知列表"""
    global _EXTEND_METRICS
    # 使用 set 去重并保持更新
    _EXTEND_METRICS = new_metrics


def process_metric(log_dict, key: str = "", metric: torch.Tensor = None, group=None, amount=-1):
    # 对于单一均值，只需把求和传入，指定数量即可，但是对于求扩展指标，需要传入原始sample mean的向量，来做基础的计算
    if not any(item in key for item in _EXTEND_METRICS):
        log_dict[key] = _mean_reduce(metric, group, amount)
    else:
        mean, std_, max_, min_ = _extend_reduce(metric, group)
        log_dict[key] = mean
        log_dict[f"{key}/max"] = max_
        log_dict[f"{key}/min"] = min_
        log_dict[f"{key}/std"] = std_


def _mean_reduce(local_vals: torch.Tensor, group, amount):
    local_sum = local_vals.sum()
    local_count = local_vals.numel()
    metric = torch.Stack([local_sum, local_count]).to(local_vals.device)
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

    global_mean = global_sum / n
    global_var = (global_sum_sq / n) - (global_mean ** 2)
    global_std = (global_var ** 0.5) if global_var > 0 else 0.0
    return global_mean, global_std, max_, min_
