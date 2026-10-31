"""What a training rank reports to run observers: the per-token values of the samples it trained on."""

from __future__ import annotations

import numpy as np
import torch

from miles.backends.training_utils.data.context_parallel import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.observers import Observers, TrainTokens

# per-token fields of which a rank holds only its context-parallel slice
_SLICED_FIELDS = ("log_probs", "rollout_log_probs", "ref_log_probs", "entropy", "ref_entropy", "advantages", "returns")


def report_train_tokens(args, *, rollout_id: int, rollout_data: dict) -> None:
    observers = Observers()
    if not observers.hears("train_tokens"):
        return
    parallel_state = get_parallel_state()
    # TP peers duplicate the DP shard and only the last PP stage computes the per-token fields
    if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
        return
    with observers.isolated("train_tokens"):
        rows = train_token_rows(
            rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
            qkv_format=args.qkv_format,
        )
        observers.train_tokens(rollout_id, rows=rows)


def train_token_rows(
    rollout_data: dict, *, rank: int, cp_rank: int, cp_size: int, qkv_format: str
) -> list[TrainTokens]:
    max_seq_lens = rollout_data.get("max_seq_lens")
    truncated = rollout_data.get("truncated")
    rows = []
    for i, sample_index in enumerate(rollout_data["sample_indices"]):
        total_length = int(rollout_data["total_lengths"][i])
        response_length = int(rollout_data["response_lengths"][i])
        positions = response_positions(
            total_length,
            response_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            qkv_format=qkv_format,
            max_seq_len=None if max_seq_lens is None else int(max_seq_lens[i]),
        )
        per_token = {field: _slice_of(rollout_data.get(field), i, expected=len(positions)) for field in _SLICED_FIELDS}
        rows.append(
            TrainTokens(
                sample_index=int(sample_index),
                rank=rank,
                response_length=response_length,
                total_length=total_length,
                reward=float(rollout_data["rewards"][i]),
                truncated=None if truncated is None else bool(truncated[i]),
                loss_mask=_numpy(rollout_data["loss_masks"][i]),
                positions=positions,
                **per_token,
            )
        )
    return rows


def response_positions(
    total_length: int, response_length: int, *, cp_rank: int, cp_size: int, qkv_format: str, max_seq_len: int | None
) -> np.ndarray:
    """The response positions a context-parallel rank's per-token values cover, in the order it holds them."""
    if cp_size == 1:
        return np.arange(response_length)
    _, _, logits_offsets, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
    )
    # the logit at absolute position p scores response position p - (prompt_length - 1)
    shift = total_length - response_length - 1
    return np.concatenate([np.arange(lo, hi) - shift for lo, hi in logits_offsets if hi > lo] or [np.arange(0)])


def _slice_of(values, i: int, *, expected: int) -> np.ndarray | None:
    if values is None:
        return None
    array = _numpy(values[i])
    # a slice that does not match the layout would misplace every value; leave the field out instead
    return array if len(array) == expected else None


def _numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach()
        return (values.float() if values.is_floating_point() else values).cpu().numpy()
    return np.asarray(values)
