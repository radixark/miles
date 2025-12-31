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

    @staticmethod
    def aggregate_and_log(
        log_dict: dict,
        args,
        rollout_id: int,
        process_group: Optional[dist.ProcessGroup] = None,
        dp_src_rank: int = 0,
        only_log_on_src: bool = True,
    ) -> Optional[dict]:
        """
        Aggregate per-rank metrics into pooled/global metrics and log them.

        Expected convention for pooled fields: callers should emit per-rank
        aggregates with suffixes: `_agg_sum`, `_agg_sumsq`, `_agg_count`,
        `_agg_min`, `_agg_max` for fields that require pooled mean/std/min/max.

        Non-aggregate scalar keys (plain numeric) will be averaged across
        ranks via all-reduce sum/mean.

        Returns the reduced dict (with keys prefixed by `rollout/`) on all
        ranks. Logging via `tracking_utils.log` happens on the DP source rank
        when `only_log_on_src` is True.
        """
        # Fast path: non-distributed -> compute locally
        if not dist.is_available() or not dist.is_initialized() or process_group is None:
            reduced: dict = {}
            # Handle aggregate bases
            agg_bases = {k[: -len("_agg_sum")] for k in log_dict.keys() if k.endswith("_agg_sum")}
            for base in agg_bases:
                s = float(log_dict.get(f"{base}_agg_sum", 0.0))
                ssq = float(log_dict.get(f"{base}_agg_sumsq", 0.0))
                cnt = int(log_dict.get(f"{base}_agg_count", 0))
                mn = float(log_dict.get(f"{base}_agg_min", float("inf")))
                mx = float(log_dict.get(f"{base}_agg_max", float("-inf")))
                if cnt > 0:
                    mean = s / cnt
                    var = ssq / cnt - mean * mean
                    if cnt >= 2:
                        var = var * (cnt / (cnt - 1))
                    std = float(max(var, 0.0) ** 0.5)
                else:
                    mean = 0.0
                    std = 0.0
                    mn = 0.0
                    mx = 0.0
                reduced[f"rollout/{base}_global_mean"] = mean
                reduced[f"rollout/{base}_global_std"] = std
                reduced[f"rollout/{base}_global_min"] = mn
                reduced[f"rollout/{base}_global_max"] = mx

            # Average non-aggregate numeric keys
            non_agg_keys = [k for k in log_dict.keys() if not any(k.startswith(b) for b in agg_bases)]
            for key in non_agg_keys:
                v = log_dict[key]
                try:
                    reduced[f"rollout/{key}"] = float(v)
                except Exception:
                    reduced[f"rollout/{key}"] = v

            # Add step if available
            reduced["rollout/step"] = compute_rollout_step(args, rollout_id)
            if not only_log_on_src:
                tracking_utils.log(args, reduced, step_key="rollout/step")
            return reduced

        # Distributed path: perform all-reduces per aggregate
        dp_size = dist.get_world_size(group=process_group)

        gathered_reduced: dict = {}

        # Find aggregate bases
        agg_bases = {k[: -len("_agg_sum")] for k in log_dict.keys() if k.endswith("_agg_sum")}

        # For each aggregate base, all-reduce sum, sumsq, count and reduce min/max
        for base in agg_bases:
            local_sum = torch.tensor(float(log_dict.get(f"{base}_agg_sum", 0.0)), dtype=torch.float64, device="cpu")
            local_sumsq = torch.tensor(
                float(log_dict.get(f"{base}_agg_sumsq", 0.0)), dtype=torch.float64, device="cpu"
            )
            local_count = torch.tensor(int(log_dict.get(f"{base}_agg_count", 0)), dtype=torch.float64, device="cpu")
            # Use CPU tensors for small reductions to avoid GPU sync
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM, group=process_group)

            total_count = int(local_count.item())
            total_sum = float(local_sum.item())
            total_sumsq = float(local_sumsq.item())

            # Reduce min/max
            local_min = torch.tensor(
                float(log_dict.get(f"{base}_agg_min", float("inf"))), dtype=torch.float64, device="cpu"
            )
            local_max = torch.tensor(
                float(log_dict.get(f"{base}_agg_max", float("-inf"))), dtype=torch.float64, device="cpu"
            )
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN, group=process_group)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=process_group)

            if total_count > 0:
                mean = total_sum / total_count
                var = total_sumsq / total_count - mean * mean
                if total_count >= 2:
                    var = var * (total_count / (total_count - 1))
                std = float(max(var, 0.0) ** 0.5)
                mn = float(local_min.item())
                mx = float(local_max.item())
            else:
                mean = 0.0
                std = 0.0
                mn = 0.0
                mx = 0.0

            gathered_reduced[f"rollout/{base}_global_mean"] = mean
            gathered_reduced[f"rollout/{base}_global_std"] = std
            gathered_reduced[f"rollout/{base}_global_min"] = mn
            gathered_reduced[f"rollout/{base}_global_max"] = mx

        # Handle non-aggregate numeric keys: all-reduce sum then divide by dp_size
        non_agg_keys = [k for k in log_dict.keys() if not any(k.startswith(b) for b in agg_bases)]
        for key in non_agg_keys:
            v = log_dict[key]
            try:
                t = torch.tensor(float(v), dtype=torch.float64, device="cpu")
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=process_group)
                gathered_reduced[f"rollout/{key}"] = float(t.item()) / float(dp_size)
            except Exception:
                # Non-numeric -> pick first-rank's value via gather_object
                vals = [None] * dp_size
                dist.gather_object(
                    log_dict[key],
                    vals if dist.get_rank() == dp_src_rank else None,
                    dst=dp_src_rank,
                    group=process_group,
                )
                if dist.get_rank() == dp_src_rank:
                    gathered_reduced[f"rollout/{key}"] = vals[0]

        # Add rollout step
        gathered_reduced["rollout/step"] = compute_rollout_step(args, rollout_id)

        # Logging only on source rank by default
        rank = dist.get_rank(group=process_group) if hasattr(dist, "get_rank") else dist.get_rank()
        if not only_log_on_src or rank == dp_src_rank:
            tracking_utils.log(args, gathered_reduced, step_key="rollout/step")

        return gathered_reduced
