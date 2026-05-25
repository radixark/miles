"""CI utilities for training backend testing."""

import logging
import math
from argparse import Namespace

import torch

logger = logging.getLogger(__name__)


def _check_upper_bound(args: Namespace, log_dict: dict[str, float], arg_name: str, metric_name: str) -> None:
    threshold = getattr(args, arg_name, None)
    if threshold is None:
        return

    assert metric_name in log_dict, f"CI check failed: {metric_name} missing with {arg_name}={threshold}; {log_dict=}"
    actual_value = log_dict[metric_name]
    assert (
        actual_value < threshold
    ), f"CI check failed: {metric_name} ({actual_value}) must be < {arg_name} ({threshold}); {log_dict=}"


def check_kl(args: Namespace, log_dict: dict[str, float], step_id: int, accumulated_step_id: int) -> None:
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        if args.multi_latent_attention:
            # TODO: mla currently have non-zero kl, need further investigation
            assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
        elif getattr(args, "lora_rank", 0) > 0:
            # LoRA weight conversion (Megatron → HF for SGLang) introduces
            # small floating-point differences, so use a relaxed threshold.
            assert abs(log_dict["train/ppo_kl"]) < 1e-8 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
        else:
            assert abs(log_dict["train/ppo_kl"]) < 1e-9 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
    if accumulated_step_id == 0 and "train/kl_loss" in log_dict and not args.use_rollout_routing_replay:
        assert abs(log_dict["train/kl_loss"]) < 1e-9, f"{log_dict=}"
    _check_upper_bound(args, log_dict, "ci_max_train_rollout_logprob_abs_diff", "train/train_rollout_logprob_abs_diff")
    _check_upper_bound(args, log_dict, "ci_max_kl_loss", "train/kl_loss")


def check_grad_norm(
    args: Namespace,
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    role: str = "actor",
    rank: int = 0,
) -> None:

    if rank != 0:
        return

    if args.ci_save_grad_norm is not None:
        ci_save_grad_norm_path = args.ci_save_grad_norm.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        torch.save(grad_norm, ci_save_grad_norm_path)

    elif args.ci_load_grad_norm is not None:
        ci_load_grad_norm_path = args.ci_load_grad_norm.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        expected_grad_norm = torch.load(ci_load_grad_norm_path, weights_only=False)
        assert math.isclose(
            grad_norm,
            expected_grad_norm,
            rel_tol=0.03,
            abs_tol=0.03,
        ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
