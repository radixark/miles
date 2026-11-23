"""CI utilities for training backend testing."""

import logging
import math
from argparse import Namespace

import torch

from miles.utils.metric_utils import strip_metrics_namespace

logger = logging.getLogger(__name__)


def check_true_on_policy_logprobs(args: Namespace, log_dict: dict[str, float], step_id: int) -> None:
    """Under TOP every training-forward log-prob equals the rollout's, bit for bit. Checked on the training forward,
    which every recipe runs (the forward-only pass is skipped when rollout log-probs are the old policy). Only the
    rollout's first optimizer step scores the weights that sampled it; later steps score updated weights."""
    log_dict = strip_metrics_namespace(log_dict, trainer_model_id=args.trainer_model_id)
    if step_id != 0 or "train/pg_loss" not in log_dict:
        return  # this rank logs no policy loss (an earlier pipeline stage)
    diff = log_dict.get("train/train_rollout_logprob_abs_diff")
    assert diff == 0.0, f"CI check failed: true_on_policy_mode is enabled, but train_rollout_logprob_abs_diff={diff}"


def check_kl(args: Namespace, log_dict: dict[str, float], step_id: int, accumulated_step_id: int) -> None:
    log_dict = strip_metrics_namespace(log_dict, trainer_model_id=args.trainer_model_id)
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        if args.backend.multi_latent_attention:
            # TODO: mla currently have non-zero kl, need further investigation
            assert abs(log_dict["train/ppo_kl"]) < 1e-8, f"{log_dict=}"
        elif args.lora_rank > 0:
            # LoRA weight conversion (Megatron → HF for SGLang) introduces
            # small floating-point differences, so use a relaxed threshold.
            assert abs(log_dict["train/ppo_kl"]) < 1e-8 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
        else:
            assert abs(log_dict["train/ppo_kl"]) < 1e-9 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
    if accumulated_step_id == 0 and "train/kl_loss" in log_dict and not args.use_rollout_routing_replay:
        assert abs(log_dict["train/kl_loss"]) < 1e-9, f"{log_dict=}"


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

    if (x := args.ci_save_grad_norm) is not None:
        ci_save_grad_norm_path = x.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        torch.save(grad_norm, ci_save_grad_norm_path)

    elif (x := args.ci_load_grad_norm) is not None:
        ci_load_grad_norm_path = x.format(
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
