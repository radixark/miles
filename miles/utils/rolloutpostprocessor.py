from typing import Optional

import torch
import torch.distributed as dist


class RolloutPostprocessor:
    """Postprocessing helpers for rollout / loss metrics.

    This class centralizes distributed, masked statistics used when
    reporting metrics such as advantage and per-token pg_loss distributions.
    """

    @staticmethod
    def compute_global_masked_stats(
        values: torch.Tensor,
        mask: torch.Tensor,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> dict:
        """
        Compute global mean, std, min and max over elements where `mask` is truthy,
        aggregating across the provided process group.

        Returns dict with keys `mean`, `std`, `min`, `max` as torch tensors on
        `values.device`.
        """
        mask_bool = mask.bool()

        local_sum = (values * mask_bool).sum()
        local_sum_sq = ((values**2) * mask_bool).sum()
        local_count = mask_bool.sum().to(dtype=torch.float32)

        stats_tensor = torch.tensor([local_sum, local_sum_sq, local_count], device=values.device, dtype=torch.float32)
        dist.all_reduce(stats_tensor, group=process_group)

        global_sum, global_sum_sq, global_count = stats_tensor

        if global_count.item() == 0:
            zero = torch.tensor(0.0, device=values.device)
            return {
                "mean": zero,
                "std": zero,
                "min": torch.tensor(float("inf"), device=values.device),
                "max": torch.tensor(float("-inf"), device=values.device),
            }

        global_mean = global_sum / global_count
        global_mean_sq = global_sum_sq / global_count
        global_var = global_mean_sq - global_mean**2

        if global_count.item() >= 2:
            bessel = global_count / (global_count - 1)
            global_var = global_var * bessel

        global_std = torch.sqrt(torch.clamp(global_var, min=0.0))

        local_max = torch.where(mask_bool, values, torch.tensor(float("-inf"), device=values.device))
        local_min = torch.where(mask_bool, values, torch.tensor(float("inf"), device=values.device))

        max_tensor = local_max.max() if local_max.numel() > 0 else torch.tensor(float("-inf"), device=values.device)
        min_tensor = local_min.min() if local_min.numel() > 0 else torch.tensor(float("inf"), device=values.device)

        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX, group=process_group)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN, group=process_group)

        return {"mean": global_mean, "std": global_std, "min": min_tensor, "max": max_tensor}

    @staticmethod
    def compute_masked_stats_safe(
        values: torch.Tensor,
        mask: torch.Tensor,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> dict:
        """
        Safe wrapper around `compute_global_masked_stats` that falls back to
        local (non-distributed) statistics when torch.distributed is not
        available or not initialized. This avoids runtime errors in contexts
        (e.g., Ray rollout workers) where distributed backend isn't set up.

        Returns the same dict format: {"mean", "std", "min", "max"}.
        """
        # If distributed isn't available/initialized, compute local masked stats.
        if not dist.is_available() or not dist.is_initialized():
            mask_bool = mask.bool()
            if mask_bool.numel() == 0 or mask_bool.sum().item() == 0:
                zero = torch.tensor(0.0, device=values.device)
                return {
                    "mean": zero,
                    "std": zero,
                    "min": torch.tensor(float("inf"), device=values.device),
                    "max": torch.tensor(float("-inf"), device=values.device),
                }

            vals = values[mask_bool]
            mean = vals.mean()
            std = vals.std(unbiased=False) if vals.numel() > 1 else torch.tensor(0.0, device=values.device)
            min_v = vals.min()
            max_v = vals.max()
            return {"mean": mean, "std": std, "min": min_v, "max": max_v}

        # Otherwise delegate to the distributed implementation
        return RolloutPostprocessor.compute_global_masked_stats(values, mask, process_group=process_group)
