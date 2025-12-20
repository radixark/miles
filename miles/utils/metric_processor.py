import logging
from typing import Optional, Callable
import torch.distributed as dist

from miles.utils import tracking_utils
import torch

logger = logging.getLogger(__name__)


class MetricProcessor:
    def __init__(self, args, extend_list=None):
        self.args = args
        if extend_list is None:
            self.extend_list = ["advantages", "pg_loss"]
        else:
            self.extend_list = extend_list

    def log_metrics(self, step_key, log_dict):
        tracking_utils.log(self.args, log_dict, step_key=step_key)

    def process_metric(self, log_dict, key: str = "", amount=0, metric: torch.Tensor = None, group=None):
        # 对于单一均值，只需把求和传入，指定数量即可，但是对于求扩展指标，需要传入原始sample mean的向量，来做基础的计算
        # todo:深入考虑对样本数量的计算和边界
        if not any(item in key for item in self.extend_list):
            log_dict[key] = _mean_reduce(metric, group, amount)
        else:
            mean, std_, max_, min_ = _extend_reduce(metric, group, amount)
            log_dict[key] = mean
            log_dict[f"{key}/max"] = max_
            log_dict[f"{key}/min"] = min_
            log_dict[f"{key}/std"] = std_


def _mean_reduce(local_vals: torch.Tensor, group, amount):
    local_sum = local_vals.sum()
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
    return local_sum / amount


def _extend_reduce(local_vals: torch.Tensor, group, amount):
    local_sum = local_vals.sum()
    local_sum_sq = (local_vals ** 2).sum()
    if local_vals.numel() > 0:
        max_ = local_vals.max()
        min_ = local_vals.min()
    else:
        max_ = torch.tensor(float("-inf"), device=local_vals.device, dtype=local_vals.dtype)
        min_ = torch.tensor(float("inf"), device=local_vals.device, dtype=local_vals.dtype)
    local_count = torch.tensor(local_vals.shape[0], dtype=torch.float32, device=local_vals.device)

    metric = torch.stack([local_sum, local_sum_sq, local_count])

    dist.all_reduce(metric, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(min_, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(max_, op=dist.ReduceOp.MAX, group=group)

    global_sum, global_sum_sq, n = metric.tolist()

    global_mean = global_sum / n
    global_var = (global_sum_sq / n) - (global_mean ** 2)
    global_std = (global_var ** 0.5) if global_var > 0 else 0.0
    return global_mean, global_std, max_, min_
