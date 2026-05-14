from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from miles.utils.types import RolloutBatch


@dataclass(frozen=True)
class LossFnInput:
    args: Namespace

    # Loss-type-dependent. Common keys: unconcat_tokens, response_lengths,
    # total_lengths, loss_masks (one per sample); max_seq_lens required when
    # qkv_format == "bshd". policy_loss adds advantages, log_probs (+ optional
    # ref_/rollout_log_probs); value_loss adds values, returns.
    batch: RolloutBatch

    # Float32. Last dim is vocab_size (policy/sft) or 1 (value). Outer shape is
    # [1, T, ...] for qkv_format="thd" (T = sum of total_lengths), or
    # [B, max_seq_len, ...] for "bshd".
    logits: torch.Tensor

    # CP-aware reducer: flat per-token tensor → scalar, sample-mean weighted by loss_masks.
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LossFnOutput:
    # Scalar with grad. UN-rescaled — loss_function applies Megatron scaling after.
    loss: torch.Tensor

    # Detached 0-d scalars for the training log. Keys per loss type:
    #   policy_loss : "loss", "pg_loss", "entropy_loss", "pg_clipfrac", "ppo_kl",
    #                 "ess_ratio"; + flag-gated "kl_loss", "ois"/"tis"/"tis_clipfrac"
    #                 /"tis_abs", "opsm_clipfrac", "train_rollout_logprob_abs_diff"
    #                 /"train_rollout_kl".
    #   value_loss  : "value_loss", "value_clipfrac"
    #   sft_loss    : "loss"
    metrics: dict[str, torch.Tensor]


class LossFunction(Protocol):
    """Signature of per-loss-type functions dispatched by `get_loss_function`."""

    def __call__(self, input: LossFnInput) -> LossFnOutput: ...
