import torch
import torch.distributed as dist
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)

class RolloutPostprocessor:
    """
    Unified pipeline for processing statistics of rollouts.
    Satisfies the requirement for a cleaner, unified way to handle metrics across backends
    right after feedforward.
    """
    def __init__(self, args: Namespace):
        self.args = args

    def compute_and_log_stats(self, log_dict, key, val_tensor, sample_means_func, cp_size, dp_group, prefix="train"):
        """
        Unified entry point for calculating global Max, Min, Mean, and Std.
        Both Megatron and FSDP backends call this to ensure consistent logging and math.
        """
        stats = compute_global_stats(val_tensor, sample_means_func, cp_size, dp_group)
        
        # Consistent naming convention for all backends
        for k, v in stats.items():
            log_dict[f"{prefix}/{key}/{k}"] = v
            
        return stats["mean"]

def compute_global_stats(val_tensor, sample_means_func, cp_size, dp_group):
    """
    Core math for global statistics. Handles the distributed reduction of 
    count, sum, sum_of_squares, max, and min to derive global Mean and Std Dev.
    """
    # 1. Compute per-sample means on this rank (CP-aware via sample_means_func)
    if val_tensor is not None:
        local_sample_means = sample_means_func(val_tensor) * cp_size
    else:
        # Fallback if sample_means are already computed and passed via lambda
        local_sample_means = sample_means_func(None)
    
    local_sample_means = local_sample_means.float().detach()
    
    if local_sample_means.numel() == 0:
        # Handle empty case
        device = torch.cuda.current_device()
        count = torch.tensor([0.0], device=device)
        sum_val = torch.tensor([0.0], device=device)
        sum_sq = torch.tensor([0.0], device=device)
        max_val = torch.tensor([-float('inf')], device=device)
        min_val = torch.tensor([float('inf')], device=device)
    else:
        count = torch.tensor([float(local_sample_means.numel())], device=local_sample_means.device)
        sum_val = local_sample_means.sum()
        sum_sq = (local_sample_means ** 2).sum()
        max_val = local_sample_means.max()
        min_val = local_sample_means.min()

    # 2. Global reduction across Data Parallel group
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(sum_val, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(max_val, op=dist.ReduceOp.MAX, group=dp_group)
    dist.all_reduce(min_val, op=dist.ReduceOp.MIN, group=dp_group)

    if count.item() == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

    mean = sum_val / count
    var = (sum_sq / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    
    return {
        "mean": mean.item(),
        "max": max_val.item(),
        "min": min_val.item(),
        "std": std.item(),
    }

def log_stats(log_dict, prefix, stats):
    """Helper for legacy logging formats."""
    for k, v in stats.items():
        log_dict[f"{prefix}/{k}"] = v