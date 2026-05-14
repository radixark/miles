from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from miles.utils.types import RolloutBatch


@dataclass(frozen=True)
class LossFnContext:
    args: Namespace

    # Loss-type-dependent. Common keys: unconcat_tokens, response_lengths,
    # total_lengths, loss_masks (one per sample); max_seq_lens required when
    # qkv_format == "bshd". policy_loss adds advantages, log_probs (+ optional
    # ref_/rollout_log_probs); value_loss adds values, returns.
    batch: RolloutBatch

    # CP-aware reducer: flat per-token tensor → scalar, sample-mean weighted by loss_masks.
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor]


class LossFunction(Protocol):
    """Per-loss-type function dispatched by `get_loss_function`.

    `logits` is float32; last dim is vocab_size (policy/sft) or 1 (value).
    Outer shape is [1, T, ...] for qkv_format="thd" (T = sum of total_lengths)
    or [B, max_seq_len, ...] for "bshd".

    Returns `(loss, metrics)`:
      * `loss`: scalar tensor with grad, un-rescaled (the dispatcher applies
        Megatron scaling on top).
      * `metrics`: dict of detached 0-d scalars, surfaced under `train/` in
        the training log / wandb.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        ctx: LossFnContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
