"""Train-side dumps of one widened rollout step, written by miles' own dump functions."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.backends.training_utils.data.context_parallel import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.metrics import debug_dump
from miles.backends.training_utils.metrics.train_dump import save_debug_train_data_for_rank
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp,
)
from miles.utils.dp_schedule import TrainParallelConfig
from miles.utils.types import Sample


@dataclass(frozen=True)
class ActorLayout:
    """The calibration run's actor parallelism and dynamic batch size (its launcher's flags)."""

    tp_size: int
    pp_size: int
    cp_size: int
    max_tokens_per_gpu: int
    # a ref model pass (--use-kl-loss) adds ref_log_probs / ref_entropy
    with_ref: bool

    def dp_size(self, train_gpus: int) -> int:
        model_parallel_gpus = self.tp_size * self.pp_size * self.cp_size
        assert train_gpus % model_parallel_gpus == 0, f"{train_gpus} train GPUs do not divide into {self} replicas"
        return train_gpus // model_parallel_gpus


class TrainDumpWriter:
    """Train dumps at the widened actor layout: one file per (DP, CP)
    coordinate, from TP rank 0 of the last PP stage like
    ``save_debug_train_data``, plus one ``policy_loss_debug`` file per microbatch."""

    def __init__(self, dump_args: Namespace, *, out: Path, train_gpus: int, layout: ActorLayout):
        self.dump_args = dump_args
        self.out = out
        self.layout = layout
        self.dp_size = layout.dp_size(train_gpus)
        self.train_parallel_config = TrainParallelConfig(
            dp_size=self.dp_size,
            cp_size=layout.cp_size,
            vpp_size=None,
            microbatch_group_size_per_vp_stage=None,
            independent_dp=False,
            supports_precomputed_schedule=True,  # megatron
        )
        # per-process counter in a real run: one per rank here
        self.policy_loss_calls_by_rank: dict[int, int] = defaultdict(int)

    def write(self, rollout_id: int, samples: list[Sample]) -> None:
        train_data = convert_samples_to_train_data(
            self.dump_args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        generator = torch.Generator().manual_seed(rollout_id)
        for dp_rank, shard in enumerate(
            split_train_data_by_dp(self.dump_args, train_data, self.train_parallel_config)
        ):
            shard = process_rollout_data_shard(self.dump_args, shard)
            per_token_fields = _compute_actor_per_token_fields(shard, generator, with_ref=self.layout.with_ref)
            layout = self.layout
            for cp_rank in range(layout.cp_size):
                # megatron rank order tp-cp-ep-dp-pp
                rank = layout.tp_size * (cp_rank + layout.cp_size * (dp_rank + self.dp_size * (layout.pp_size - 1)))
                cp_shard = _slice_shard_for_cp_rank(shard, per_token_fields, cp_rank=cp_rank, cp_size=layout.cp_size)
                save_debug_train_data_for_rank(
                    self.dump_args,
                    rollout_id=rollout_id,
                    rollout_data=cp_shard,
                    rank=rank,
                    cp_rank=cp_rank,
                    cp_size=layout.cp_size,
                )
                self._write_policy_loss_debug(cp_shard, rank=rank)

    def _write_policy_loss_debug(self, cp_shard: dict, *, rank: int) -> None:
        with _policy_loss_debug_rank_context(rank, call=self.policy_loss_calls_by_rank[rank]):
            for rows in cp_shard["micro_batch_indices"]:
                columns = {
                    key: [cp_shard[key][row] for row in rows]
                    for key in ("log_probs", "rollout_log_probs", "advantages")
                }
                debug_dump.maybe_dump_policy_loss_debug(
                    args=SimpleNamespace(dump_details=str(self.out)),
                    batch=dict(
                        total_lengths=[cp_shard["total_lengths"][row] for row in rows],
                        response_lengths=[cp_shard["response_lengths"][row] for row in rows],
                    ),
                    train_log_probs=columns["log_probs"],
                    old_log_probs=[values.clone() for values in columns["log_probs"]],
                    rollout_log_probs=columns["rollout_log_probs"],
                    advantages=columns["advantages"],
                    local_loss_masks=[torch.ones_like(values) for values in columns["log_probs"]],
                    # per local token, as the loss computes them
                    ppo_kl=torch.zeros(sum(len(values) for values in columns["log_probs"])),
                    pg_loss=torch.zeros(sum(len(values) for values in columns["log_probs"])),
                )
            self.policy_loss_calls_by_rank[rank] = debug_dump._POLICY_LOSS_DUMP_COUNTER


@contextmanager
def _policy_loss_debug_rank_context(rank: int, *, call: int):
    """Run ``maybe_dump_policy_loss_debug`` as if inside ``rank``'s process."""
    saved = debug_dump.get_parallel_state, debug_dump.dist, debug_dump._POLICY_LOSS_DUMP_COUNTER
    debug_dump.get_parallel_state = lambda: SimpleNamespace(tp=SimpleNamespace(rank=0))
    debug_dump.dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: rank)
    debug_dump._POLICY_LOSS_DUMP_COUNTER = call
    try:
        yield
    finally:
        debug_dump.get_parallel_state, debug_dump.dist, debug_dump._POLICY_LOSS_DUMP_COUNTER = saved


def _compute_actor_per_token_fields(
    shard: dict, generator: torch.Generator, *, with_ref: bool
) -> dict[str, list[torch.Tensor]]:
    """Full-response per-token fields; values are synthetic (they need the model's forward pass)."""
    rollout_log_probs = [torch.tensor(values, dtype=torch.float32) for values in shard["rollout_log_probs"]]
    advantages = [
        torch.full((len(values),), float(reward))
        for values, reward in zip(rollout_log_probs, shard["rewards"], strict=True)
    ]
    per_token_fields = dict(
        rollout_log_probs=rollout_log_probs,
        log_probs=[values + 0.01 * torch.randn(len(values), generator=generator) for values in rollout_log_probs],
        entropy=[torch.rand(len(values), generator=generator) for values in rollout_log_probs],
        advantages=advantages,
        returns=advantages,  # GRPO: the same tensors, saved once
    )
    if with_ref:
        per_token_fields["ref_log_probs"] = [values.clone() for values in per_token_fields["log_probs"]]
        per_token_fields["ref_entropy"] = [values.clone() for values in per_token_fields["entropy"]]
    return per_token_fields


def _slice_shard_for_cp_rank(
    shard: dict, per_token_fields: dict[str, list[torch.Tensor]], *, cp_rank: int, cp_size: int
) -> dict:
    """The shard as ``get_rollout_data`` and the forward passes leave it on
    ``cp_rank``: tokens and loss masks full length, per-token fields zigzag-sliced."""
    cp_shard = {key: value for key, value in shard.items() if key not in per_token_fields}
    cp_shard["tokens"] = [torch.tensor(tokens, dtype=torch.long) for tokens in shard["tokens"]]
    cp_shard["loss_masks"] = [torch.tensor(mask, dtype=torch.int) for mask in shard["loss_masks"]]
    if "rollout_mask_sums" in shard:
        cp_shard["rollout_mask_sums"] = torch.tensor(shard["rollout_mask_sums"], dtype=torch.float32)
    sliced_by_field_id = {}  # fields that are one object (returns is advantages) stay one after slicing
    for key, per_sample in per_token_fields.items():
        if id(per_sample) not in sliced_by_field_id:
            sliced_by_field_id[id(per_sample)] = [
                _slice_response_for_cp_rank(values, total_length, response_length, cp_rank=cp_rank, cp_size=cp_size)
                for values, total_length, response_length in zip(
                    per_sample, shard["total_lengths"], shard["response_lengths"], strict=True
                )
            ]
        cp_shard[key] = sliced_by_field_id[id(per_sample)]
    return cp_shard


def _slice_response_for_cp_rank(
    values: torch.Tensor, total_length: int, response_length: int, *, cp_rank: int, cp_size: int
) -> torch.Tensor:
    if cp_size == 1:
        return values
    _, _, logits_offsets, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, "thd", None, cp_rank=cp_rank, cp_size=cp_size
    )
    first_logit = total_length - response_length - 1
    return torch.cat([values[start - first_logit : end - first_logit] for start, end in logits_offsets])
