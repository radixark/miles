import logging
from argparse import Namespace
from math import isclose

import numpy as np
import psutil
import torch
import torch.distributed as dist

from miles.utils import train_metric_utils
from miles.utils.flops_utils import fwd_tflops_per_gpu
from miles.utils.ft_utils.process_group_utils import MultiPGUtil
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.types import RolloutBatch

from ...utils.tracking_utils import tracking
from .cp_utils import get_sum_of_sample_mean
from .data import DataIterator
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)

_MULTI_TURN_REDUCTION_BY_KEY = {
    "raw_response_length/response_length_max": "max",
    "raw_response_length/response_length_min": "min",
    "wo_obs_response_length/response_length_max": "max",
    "wo_obs_response_length/response_length_min": "min",
    "multi_turn_metric/round_number_max": "max",
    "multi_turn_metric/round_number_min": "min",
}


def reduce_gathered_log_dict(
    gathered: list[dict],
    dp_size: int,
    reduction_by_key: dict[str, str] | None = None,
) -> dict[str, float]:
    """Reduce already-gathered per-rank metrics without adding another collective.

    ``(sum, count)`` tuples reduce as ``Σsum / Σcount``. Scalars reduce by the
    reduction named in ``reduction_by_key`` ("mean", "min" or "max");
    unspecified keys reduce by mean. Metric names do not implicitly determine
    their reduction semantics. Rank-local extrema must be reduced as extrema:
    averaging per-rank maxima under-reports the global maximum (and
    over-reports the global minimum).
    """
    if not gathered:
        return {}

    expected_keys = gathered[0].keys()
    if reduction_by_key is not None:
        for rank, rank_metrics in enumerate(gathered[1:], start=1):
            if rank_metrics.keys() != expected_keys:
                raise ValueError(
                    f"Metric keys differ across ranks: rank 0={list(expected_keys)}, "
                    f"rank {rank}={list(rank_metrics.keys())}."
                )
    reduction_by_key = reduction_by_key or {}

    reduced: dict[str, float] = {}
    for key in expected_keys:
        values = [d[key] for d in gathered]
        first = values[0]
        reduction = reduction_by_key.get(key, "mean")
        if reduction not in ("mean", "min", "max"):
            raise ValueError(f"Unsupported metric reduction {reduction!r} for {key!r}.")
        if isinstance(first, tuple) and len(first) == 2:
            total_sum = sum(v[0] for v in values)
            total_count = sum(v[1] for v in values)
            reduced[key] = total_sum / total_count if total_count else 0.0
        elif reduction == "mean":
            reduced[key] = sum(values) / dp_size
        elif reduction == "min":
            reduced[key] = min(values)
        else:
            reduced[key] = max(values)
    return reduced


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
    reduction_by_key: dict[str, str] | None = None,
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce on the DP source rank, and log.

    ``(sum, count)`` tuple values reduce as ``Σsum / Σcount``; scalar keys
    reduce by mean unless `reduction_by_key` explicitly selects "min" or
    "max" for them. Returns the reduced dict on the DP source rank; returns
    None on others.
    """

    parallel_state = get_parallel_state()

    pg = parallel_state.effective_dp_cp
    log_structured(logger.info, op="cross_cell", phase="start", kind="log_gather", rank=pg.rank)
    try:
        gathered_log_dict = MultiPGUtil.gather_object(
            obj=log_dict,
            groups_inner_to_outer=pg.gloo_groups_inner_to_outer,
        )
        log_structured(logger.info, op="cross_cell", phase="end", kind="log_gather", rank=pg.rank, success=True)
    except RuntimeError:
        log_structured(
            logger.warning,
            op="cross_cell",
            phase="end",
            kind="log_gather",
            rank=pg.rank,
            success=False,
            degraded=True,
            exc_info=True,
        )
        return None

    if pg.rank == 0:
        reduced = reduce_gathered_log_dict(gathered_log_dict, pg.size, reduction_by_key)
        reduced_log_dict = {f"{metric_name}/{key}": value for key, value in reduced.items()}
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        return None


def aggregate_forward_results(
    forward_data_store: list[dict[str, list]],
    data_iterator: DataIterator,
    args: Namespace,
    store_prefix: str = "",
) -> dict[str, list]:
    rollout_data = {}
    if not forward_data_store:
        return rollout_data

    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for batch_result in forward_data_store:
            assert isinstance(batch_result[key], list), f"Expected list for key {key}, got {type(batch_result[key])}"
            values += batch_result[key]

        # Handle dynamic batch size: restore original order
        if args.use_dynamic_batch_size and hasattr(data_iterator, "micro_batch_indices"):
            origin_values = [None] * len(values)
            origin_indices = sum(data_iterator.micro_batch_indices, [])
            for value, origin_index in zip(values, origin_indices, strict=False):
                origin_values[origin_index] = value
            values = origin_values

        rollout_data[key] = values

    return rollout_data


def log_rollout_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        cp_size = parallel_state.cp.size
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        # per-rollout-mean count share: num_rollouts / dp (None = legacy local count)
        rollout_count_share = None
        if (num_rollouts := rollout_data.get("num_rollouts")) is not None:
            rollout_count_share = sum(num_rollouts) / parallel_state.intra_dp.size

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_routed_experts",
                "rollout_indexer_topk",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "witness_ids",
                "weight_versions",
                "metadata",
                "num_microbatches",
                "micro_batch_indices",
                "num_rollouts",
                "n_adapters",
                "adapter_slots",
                "step_slots",
                "step_adapter_names",
                "step_adapter_batch_sizes",
                "prompt_group_sizes",
            ]:
                continue
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    count = len(val)
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    tensor = torch.cat(val).clone().detach()
                    if tensor.device != loss_masks[0].device:
                        tensor = tensor.to(loss_masks[0].device)
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                        "entropy",
                    ]:
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                            denominators=rollout_data.get("rollout_mask_sums", None),
                        )
                        per_rank_sum = cp_size * sum_of_sample_mean(tensor)
                        if rollout_count_share is not None:
                            count = rollout_count_share
                    else:
                        per_rank_sum = tensor.mean() * cp_size * count
                    log_dict[key] = (per_rank_sum.item(), count)
                else:
                    # Flatten nested lists (e.g. list of lists from async rollout)
                    flat = val
                    if isinstance(val[0], (list, tuple)):
                        flat = [x for sublist in val for x in sublist]
                    # Skip non-numeric values (e.g. strings from async rollout metadata)
                    if flat and not isinstance(flat[0], (int, float)):
                        continue
                    log_dict[key] = (sum(flat), len(flat))
            elif isinstance(val, torch.Tensor):
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and not args.ci_disable_logprobs_checker and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # When R3 (rollout routing replay) is enabled, ref model does not use R3
                # so log_probs and ref_log_probs may diverge; use a relaxed tolerance.
                # When --sglang-config deploys multiple models, the heavier offload/onload
                # cycle can amplify flash-attention non-determinism; use 1e-8.
                # The default branch also covers larger TP/CP/EP variants (e.g. stage-c-long
                # test_qwen2.5_0.5B_gsm8k.py on 8 GPUs hit ~3.7e-9 diff in CI), so use 1e-8
                # rather than the previous 3e-9 to absorb BF16 reduction noise across configs.
                if args.use_rollout_routing_replay:
                    # lop diff w/ w/o r3 is very big
                    abs_tol = 5e-3
                elif getattr(args, "sglang_config", None) is not None:
                    abs_tol = 1e-8
                else:
                    abs_tol = 1e-8
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/ref_log_probs"], abs_tol=abs_tol
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != ref_log_probs ({reduced_log_dict['rollout/ref_log_probs']})"
            if "rollout/log_probs" in reduced_log_dict and "rollout/rollout_log_probs" in reduced_log_dict:
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/rollout_log_probs"], abs_tol=0.03
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != rollout_log_probs ({reduced_log_dict['rollout/rollout_log_probs']})"
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.7

        if args.ci_test and args.true_on_policy_mode and not args.ci_disable_logprobs_checker:
            assert log_dict["log_probs"] == log_dict["rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout_log_probs']})"
            )

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
            cp_size = parallel_state.cp.size
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # per-sample mean over the correct subset, not per-rollout
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, denominators=None
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data(
            "multi_turn",
            args,
            rollout_id,
            log_dict,
            reduction_by_key=_MULTI_TURN_REDUCTION_BY_KEY,
        )


def log_perf_data(rollout_id: int, args: Namespace, extra_metrics: dict | None = None) -> None:
    parallel_state = get_parallel_state()
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            parallel_state.tp.rank == 0
            and parallel_state.is_pp_last_stage
            and parallel_state.effective_dp_cp.rank == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: fwd_tflops_per_gpu(seq_lens, args, dist.get_world_size()),
        extra_metrics=extra_metrics,
    )


def log_cpu_memory(rollout_id: int, args: Namespace, label: str) -> None:
    """Log current system CPU memory usage to wandb/tensorboard.

    Caller is responsible for ensuring this runs on a single rank only.
    """

    cpu_mem_gb = psutil.virtual_memory().used / 1e9
    step = compute_rollout_step(args, rollout_id)
    logger.info(f"[CPU memory] {label}: {cpu_mem_gb:.2f} GB (rollout_id={rollout_id}, step={step})")
    tracking.log(
        args,
        {f"perf/cpu_memory_{label}_gb": cpu_mem_gb, "rollout/step": step},
        step_key="rollout/step",
    )


def aggregate_train_losses(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
    num_rollouts: int | None = None,
) -> dict[str, float]:
    """Aggregate loss metrics across micro-batches.

    Sums loss values across all micro-batches, performs all-reduce across
    the data-parallel group, and computes per-sample/token averages.

    Args:
        losses_reduced: List of log_dict from each micro-batch.
            Each log_dict has format: {"keys": list[str], "values": torch.Tensor}
        num_rollouts: report per-rollout means — divide every metric by this
            step's rollout count (no CP factor; CP-chunked numerators reconstruct
            exactly once under the DP*CP all-reduce). None keeps the legacy
            reduction: divide by the all-reduced ``values[0]`` count, cancelled
            by ``cp_size``.

    Returns:
        Dictionary mapping metric names to averaged values.
    """
    parallel_state = get_parallel_state()
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]

    values = None
    for log_dict in losses_reduced:
        if values is None:
            values = log_dict["values"].clone()
        else:
            values += log_dict["values"]

    assert len(keys) + 1 == values.numel(), f"Expected {len(keys) + 1} values, got {values.numel()}"

    for group in parallel_state.effective_dp_cp.groups_inner_to_outer:
        MultiPGUtil.all_reduce(values, [group], op=dist.ReduceOp.SUM)

    loss_reduced = {}
    values = values.tolist()
    if num_rollouts is not None:
        num_samples_or_tokens = num_rollouts
        cp_factor = 1
    else:
        num_samples_or_tokens = values[0]
        cp_factor = parallel_state.cp.size

    for key, value in zip(keys, values[1:], strict=False):
        loss_reduced[key] = value * cp_factor / num_samples_or_tokens

    return loss_reduced


def log_train_step(
    args: Namespace,
    loss_dict: dict[str, float],
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    role: str = "actor",
    extra_metrics: dict[str, float] | None = None,
    should_log: bool | None = None,
) -> dict[str, float]:
    """Log training metrics for one step.

    Formats loss metrics, gradient norm, and extra metrics (e.g., learning rates, MTP loss) for tracking.

    Args:
        args: Configuration.
        loss_dict: Dictionary of loss metrics from aggregate_train_losses.
        grad_norm: Global gradient L2 norm before clipping.
        rollout_id: Rollout ID.
        step_id: Step ID within the rollout.
        num_steps_per_rollout: Total number of steps per rollout.
        role: Role name (e.g., "actor", "critic").
        extra_metrics: Optional extra metrics to log (e.g., learning rates, MTP loss).
        should_log: Optional override for logging condition. If None, uses rank == 0.

    Returns:
        The formatted log_dict (for CI tests or other uses).
    """
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    role_tag = "" if role == "actor" else f"{role}-"

    log_dict_out = {
        f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
        for key, val in loss_dict.items()
    }
    log_dict_out[f"train/{role_tag}grad_norm"] = float(grad_norm)

    if extra_metrics:
        for key, val in extra_metrics.items():
            log_dict_out[f"train/{role_tag}{key}"] = val

    log_dict_out["train/step"] = accumulated_step_id

    if should_log is None:
        should_log = dist.get_rank() == 0

    if should_log:
        tracking.log(args, log_dict_out, step_key="train/step")
        logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict_out}")

    return log_dict_out
