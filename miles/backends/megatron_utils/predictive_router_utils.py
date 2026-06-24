"""Packing + transport of recorded predictive microbatches between the
forward (RECORD) pass and the training (COMPUTE) pass of Predictive
Routing Replay (PR²).

A ``RecordedPredictiveMicrobatch`` carries the per-layer cached router
inputs and old-router logits captured during the rollout-side forward,
plus sub-sampling masks and token-count metadata. It is the payload
``model.forward_only(...)`` produces and the patched router consumes.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

PREDICTIVE_STORAGE_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

@dataclass(frozen=True)
class RecordedPredictiveMicrobatch:
    old_inputs_concat: torch.Tensor | None
    old_logits_concat: torch.Tensor | None
    valid_mask: torch.Tensor
    sampled_indices: list[int]
    sample_lengths: list[int]
    total_token_count: int
    predictive_loss_scale: float
    original_total_tokens: int = 0
    selected_total_tokens: int = 0
    selected_sample_lengths: list[int] = field(default_factory=list)

    @property
    def has_valid_samples(self) -> bool:
        return self.old_inputs_concat is not None and self.old_logits_concat is not None and bool(self.sampled_indices)


def predictive_storage_dtype_to_torch_dtype(storage_dtype: str) -> torch.dtype:
    if storage_dtype not in PREDICTIVE_STORAGE_DTYPE_MAP:
        raise ValueError(
            f"Unsupported predictive storage dtype: {storage_dtype}. "
            f"Expected one of {tuple(PREDICTIVE_STORAGE_DTYPE_MAP)}."
        )
    return PREDICTIVE_STORAGE_DTYPE_MAP[storage_dtype]


def _as_list(values):
    if isinstance(values, np.ndarray):
        return list(values)
    return values


def _to_cpu_storage_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().contiguous()
    if value.device.type == "cpu":
        if torch.cuda.is_available() and not value.is_pinned():
            return value.pin_memory()
        return value
    cpu_value = torch.empty_like(value, device="cpu", pin_memory=True)
    cpu_value.copy_(value, non_blocking=False)
    return cpu_value


def _get_local_packed_token_count(
    *,
    total_length: int,
    parallel_state,
    qkv_format: str,
    max_seq_len: int | None = None,
) -> int:
    cp_size = int(getattr(parallel_state, "cp_size", 1))
    if qkv_format == "bshd":
        if max_seq_len is None:
            raise ValueError("max_seq_len must be provided when qkv_format='bshd'.")
        base_length = int(max_seq_len)
    elif qkv_format == "thd":
        base_length = int(total_length)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    if cp_size <= 1:
        return base_length

    chunk_size = (base_length + 2 * cp_size - 1) // (2 * cp_size)
    return 2 * chunk_size


def _select_predictive_sample_indices(
    *,
    sample_lengths: Sequence[int],
    downsample_batch_size: int | None = None,
    max_len_limit: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    valid_indices = [index for index, sample_length in enumerate(sample_lengths) if int(sample_length) > 0]
    if downsample_batch_size is None or downsample_batch_size >= len(valid_indices):
        return list(valid_indices)

    if max_len_limit is None:
        filtered_indices = list(valid_indices)
    else:
        filtered_indices = [index for index in valid_indices if int(sample_lengths[index]) <= max_len_limit]

    if len(filtered_indices) >= downsample_batch_size:
        perm = torch.randperm(len(filtered_indices), generator=generator)[:downsample_batch_size].tolist()
        return sorted(filtered_indices[idx] for idx in perm)

    sampled_indices = sorted(valid_indices, key=lambda index: (int(sample_lengths[index]), index))[:downsample_batch_size]
    sampled_indices.sort()
    return sampled_indices


def build_local_predictive_sample_lengths(
    *,
    total_lengths: Sequence[int],
    parallel_state,
    qkv_format: str = "thd",
    max_seq_lens: Sequence[int] | None = None,
    allgather_cp: bool = False,
) -> list[int]:
    if allgather_cp:
        raise ValueError("Predictive routing replay does not support allgather_cp layout.")

    total_lengths = [int(length) for length in _as_list(total_lengths)]
    if max_seq_lens is not None:
        max_seq_lens = [int(length) for length in _as_list(max_seq_lens)]
        if len(max_seq_lens) != len(total_lengths):
            raise ValueError(
                f"max_seq_lens length {len(max_seq_lens)} != total_lengths length {len(total_lengths)}"
            )

    sample_lengths = []
    for sample_idx, total_length in enumerate(total_lengths):
        max_seq_len = None if max_seq_lens is None else max_seq_lens[sample_idx]
        sample_lengths.append(
            _get_local_packed_token_count(
                total_length=total_length,
                parallel_state=parallel_state,
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
            )
        )
    return sample_lengths


def build_sampled_token_mask(
    *,
    sample_lengths: Sequence[int],
    sampled_token_counts: dict[int, int],
    total_token_count: int,
) -> torch.Tensor:
    valid_mask = torch.zeros(total_token_count, dtype=torch.bool)

    offset = 0
    for sample_idx, sample_length in enumerate(sample_lengths):
        sample_length = int(sample_length)
        next_offset = offset + sample_length
        if next_offset > total_token_count:
            raise ValueError(
                f"sample_lengths sum exceeded total_token_count: offset={offset}, sample_length={sample_length}, "
                f"total_token_count={total_token_count}"
            )
        keep_count = int(sampled_token_counts.get(sample_idx, 0))
        if keep_count > sample_length:
            raise ValueError(
                f"sampled keep_count exceeded sample_length: sample_idx={sample_idx}, "
                f"keep_count={keep_count}, sample_length={sample_length}"
            )
        if keep_count > 0:
            valid_mask[offset : offset + keep_count] = True
        offset = next_offset

    return valid_mask


def _allocate_balanced_keep_counts(lengths: Sequence[int], max_tokens: int | None) -> list[int]:
    lengths = [int(length) for length in lengths]
    total_tokens = sum(lengths)
    if max_tokens is None or int(max_tokens) <= 0 or total_tokens <= int(max_tokens):
        return list(lengths)

    keep_counts = [0 for _ in lengths]
    remaining_lengths = list(lengths)
    remaining_tokens = int(max_tokens)
    active_indices = [idx for idx, length in enumerate(remaining_lengths) if length > 0]

    while remaining_tokens > 0 and active_indices:
        per_sample_share = max(1, remaining_tokens // len(active_indices))
        next_active_indices = []
        for idx in active_indices:
            if remaining_tokens <= 0:
                break
            take = min(remaining_lengths[idx], per_sample_share, remaining_tokens)
            if take > 0:
                keep_counts[idx] += take
                remaining_lengths[idx] -= take
                remaining_tokens -= take
            if remaining_lengths[idx] > 0:
                next_active_indices.append(idx)
        active_indices = next_active_indices

    return keep_counts


def pack_recorded_predictive_microbatch(
    *,
    recorded_old_inputs: Sequence[torch.Tensor],
    recorded_old_logits: Sequence[torch.Tensor],
    total_lengths: Sequence[int],
    parallel_state,
    qkv_format: str = "thd",
    max_seq_lens: Sequence[int] | None = None,
    allgather_cp: bool = False,
    downsample_batch_size: int | None = None,
    max_len_limit: int | None = None,
    max_total_tokens: int | None = None,
    storage_dtype: str = "bf16",
    generator: torch.Generator | None = None,
) -> RecordedPredictiveMicrobatch:
    if len(recorded_old_inputs) != len(recorded_old_logits):
        raise ValueError(
            f"recorded_old_inputs length {len(recorded_old_inputs)} != recorded_old_logits length {len(recorded_old_logits)}"
        )
    if not recorded_old_inputs:
        empty_mask = torch.zeros(0, dtype=torch.bool)
        return RecordedPredictiveMicrobatch(
            old_inputs_concat=None,
            old_logits_concat=None,
            valid_mask=empty_mask,
            sampled_indices=[],
            sample_lengths=[],
            total_token_count=0,
            predictive_loss_scale=1.0,
        )

    token_count = int(recorded_old_inputs[0].shape[0])
    for layer_idx, (old_input, old_logit) in enumerate(zip(recorded_old_inputs, recorded_old_logits, strict=True)):
        if old_input.shape[0] != token_count or old_logit.shape[0] != token_count:
            raise ValueError(
                "Recorded predictive tensors must have aligned token counts across routers. "
                f"Layer {layer_idx}: inputs={old_input.shape}, logits={old_logit.shape}, token_count={token_count}"
            )

    sample_lengths = build_local_predictive_sample_lengths(
        total_lengths=total_lengths,
        parallel_state=parallel_state,
        qkv_format=qkv_format,
        max_seq_lens=max_seq_lens,
        allgather_cp=allgather_cp,
    )
    consumed_token_count = sum(sample_lengths)
    if consumed_token_count > token_count:
        raise ValueError(
            f"Local sample lengths sum {consumed_token_count} exceeds recorded token count {token_count}."
        )

    sampled_indices = _select_predictive_sample_indices(
        sample_lengths=sample_lengths,
        downsample_batch_size=downsample_batch_size,
        max_len_limit=max_len_limit,
        generator=generator,
    )
    offset = 0
    sample_ranges = []
    for sample_length in sample_lengths:
        next_offset = offset + sample_length
        sample_ranges.append((offset, next_offset))
        offset = next_offset

    original_total_tokens = sum(int(sample_lengths[sample_idx]) for sample_idx in sampled_indices)
    sampled_original_lengths = [int(sample_lengths[sample_idx]) for sample_idx in sampled_indices]
    sampled_keep_counts = _allocate_balanced_keep_counts(sampled_original_lengths, max_total_tokens)
    sampled_token_counts = {
        int(sample_idx): int(keep_count)
        for sample_idx, keep_count in zip(sampled_indices, sampled_keep_counts, strict=True)
        if int(keep_count) > 0
    }
    selected_total_tokens = sum(int(keep_count) for keep_count in sampled_keep_counts)

    valid_mask = build_sampled_token_mask(
        sample_lengths=sample_lengths,
        sampled_token_counts=sampled_token_counts,
        total_token_count=token_count,
    )
    predictive_loss_scale = 1.0
    if original_total_tokens > 0:
        predictive_loss_scale = min(1.0, float(selected_total_tokens) / float(original_total_tokens))

    if not sampled_token_counts:
        return RecordedPredictiveMicrobatch(
            old_inputs_concat=None,
            old_logits_concat=None,
            valid_mask=_to_cpu_storage_tensor(valid_mask),
            sampled_indices=sampled_indices,
            sample_lengths=sample_lengths,
            total_token_count=token_count,
            predictive_loss_scale=predictive_loss_scale,
            original_total_tokens=original_total_tokens,
            selected_total_tokens=selected_total_tokens,
            selected_sample_lengths=[int(length) for length in sampled_keep_counts],
        )

    target_dtype = predictive_storage_dtype_to_torch_dtype(storage_dtype)
    sampled_inputs_cpu = []
    sampled_logits_cpu = []
    for sample_idx in sampled_indices:
        keep_count = int(sampled_token_counts.get(sample_idx, 0))
        if keep_count <= 0:
            continue
        start_idx, end_idx = sample_ranges[sample_idx]
        end_idx = start_idx + keep_count
        sample_input = torch.stack(
            [old_input[start_idx:end_idx].detach() for old_input in recorded_old_inputs],
            dim=1,
        )
        sample_logit = torch.stack(
            [old_logit[start_idx:end_idx].detach() for old_logit in recorded_old_logits],
            dim=1,
        )
        if sample_input.dtype != target_dtype:
            sample_input = sample_input.to(target_dtype)
        if sample_logit.dtype != target_dtype:
            sample_logit = sample_logit.to(target_dtype)
        sampled_inputs_cpu.append(_to_cpu_storage_tensor(sample_input))
        sampled_logits_cpu.append(_to_cpu_storage_tensor(sample_logit))

    old_inputs_concat = torch.cat(sampled_inputs_cpu, dim=0)
    old_logits_concat = torch.cat(sampled_logits_cpu, dim=0)

    return RecordedPredictiveMicrobatch(
        old_inputs_concat=old_inputs_concat,
        old_logits_concat=old_logits_concat,
        valid_mask=_to_cpu_storage_tensor(valid_mask),
        sampled_indices=sampled_indices,
        sample_lengths=sample_lengths,
        total_token_count=token_count,
        predictive_loss_scale=predictive_loss_scale,
        original_total_tokens=original_total_tokens,
        selected_total_tokens=selected_total_tokens,
        selected_sample_lengths=[int(length) for length in sampled_keep_counts],
    )
