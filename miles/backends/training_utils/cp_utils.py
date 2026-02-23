from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .parallel import ParallelState


@dataclass(frozen=True)
class CPSliceSpec:
    """Per-sample descriptor for extracting response logits/tokens on this CP rank.

    Both fields are non-empty tuples of half-open [start, end) intervals.
    logits_slices indexes into the flat logits tensor (shared across all
    samples on this rank).  token_slices indexes into the per-sample
    unconcat_tokens[i] tensor.

    Empty contributions use degenerate (0, 0) slices that produce
    zero-length tensors still connected to the source through autograd.
    """

    logits_slices: tuple[tuple[int, int], ...]
    token_slices: tuple[tuple[int, int], ...]

    @property
    def local_len(self) -> int:
        """Total number of response tokens on this rank for this sample."""
        return sum(e - s for s, e in self.logits_slices)

    def to_response_slices(self, prompt_len: int) -> tuple[tuple[int, int], ...]:
        """Convert token_slices to response-space offsets.

        Useful for indexing into loss_mask or response-space log_prob
        tensors.  Degenerate spans are normalized to (0, 0).
        """
        return tuple((s - prompt_len, e - prompt_len) if s != e else (0, 0) for s, e in self.token_slices)


def compute_cp_slice_specs(
    *,
    parallel_state: ParallelState,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    chunk_size: int | None = None,
) -> list[CPSliceSpec]:
    """Compute per-sample CPSliceSpec for every sample in the batch.

    All CP-strategy-specific logic (contiguous vs zigzag vs none, thd vs bshd,
    the next-token-prediction -1 shift) is absorbed here.  The returned
    specs contain ready-to-use coordinates — no further arithmetic needed.

    Args:
        parallel_state: Parallel topology (cp_rank, cp_size, cp_slicing).
        total_lengths: Total sequence length (prompt + response) per sample.
        response_lengths: Response-only length per sample.
        qkv_format: "thd" (packed) or "bshd" (padded).
        max_seq_lens: Per-sample padded length; required when qkv_format="bshd".
        chunk_size: For contiguous CP, the size of each rank's local chunk.
            Computed from total_lengths if not provided.

    Returns:
        One CPSliceSpec per sample.
    """
    cp_size = parallel_state.cp_size
    cp_rank = parallel_state.cp_rank

    specs: list[CPSliceSpec] = []

    if cp_size == 1:

        end = 0

        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            prompt_len = total_length - response_length

            if qkv_format == "bshd":
                sample_base = max_seq_lens[i] * i
            else:
                sample_base = end

            logits_start = sample_base + prompt_len - 1
            logits_end = sample_base + total_length - 1

            if logits_start < logits_end:
                logits_slice = (logits_start, logits_end)
                token_slice = (prompt_len, total_length)
            else:
                logits_slice = (0, 0)
                token_slice = (0, 0)

            specs.append(CPSliceSpec(logits_slices=(logits_slice,), token_slices=(token_slice,)))

            if qkv_format != "bshd":
                end += total_length

        return specs

    if parallel_state.uses_contiguous_cp:
        local_start = cp_rank * chunk_size
        local_end = (cp_rank + 1) * chunk_size

        seq_start = 0

        for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
            prompt_len = total_length - response_length

            logits_global_start = seq_start + prompt_len - 1
            logits_global_end = seq_start + total_length - 1

            intersect_start = max(logits_global_start, local_start)
            intersect_end = min(logits_global_end, local_end)

            if intersect_start < intersect_end:
                logits_slice = (intersect_start - local_start, intersect_end - local_start)
                resp_offset_start = intersect_start - logits_global_start
                resp_offset_end = intersect_end - logits_global_start
                token_slice = (prompt_len + resp_offset_start, prompt_len + resp_offset_end)
            else:
                logits_slice = (0, 0)
                token_slice = (0, 0)

            specs.append(CPSliceSpec(logits_slices=(logits_slice,), token_slices=(token_slice,)))
            seq_start += total_length

        return specs

    else:
        flat_offset = 0

        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            prompt_len = total_length - response_length
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            if qkv_format == "thd":
                sample_chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
            else:
                assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
                sample_chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

            chunk_0_start = cp_rank * sample_chunk_size
            chunk_1_start = (2 * cp_size - cp_rank - 1) * sample_chunk_size

            raw_logits_0 = (
                max(chunk_0_start, prompt_len - 1),
                min(chunk_0_start + sample_chunk_size, total_length - 1),
            )
            raw_logits_1 = (
                max(chunk_1_start, prompt_len - 1),
                min(chunk_1_start + sample_chunk_size, total_length - 1),
            )

            if raw_logits_0[0] < raw_logits_0[1]:
                flat_logits_0 = (
                    flat_offset + raw_logits_0[0] - chunk_0_start,
                    flat_offset + raw_logits_0[1] - chunk_0_start,
                )
                token_0 = (raw_logits_0[0] + 1, raw_logits_0[1] + 1)
            else:
                flat_logits_0 = (0, 0)
                token_0 = (0, 0)

            if raw_logits_1[0] < raw_logits_1[1]:
                flat_logits_1 = (
                    flat_offset + sample_chunk_size + raw_logits_1[0] - chunk_1_start,
                    flat_offset + sample_chunk_size + raw_logits_1[1] - chunk_1_start,
                )
                token_1 = (raw_logits_1[0] + 1, raw_logits_1[1] + 1)
            else:
                flat_logits_1 = (0, 0)
                token_1 = (0, 0)

            specs.append(CPSliceSpec(logits_slices=(flat_logits_0, flat_logits_1), token_slices=(token_0, token_1)))
            flat_offset += 2 * sample_chunk_size

        return specs


def get_packed_batch_offsets_with_allgather_cp(
    total_lengths: list[int],
    response_lengths: list[int],
    parallel_state: ParallelState,
    chunk_size: int,
) -> list[dict]:
    """
    Calculate per-sequence offsets within a packed batch for all-gather CP.

    Returns a list of dicts, one per sequence, with:
    - 'local_logits_start': start of valid logits within local chunk (or -1 if none)
    - 'local_logits_end': end of valid logits within local chunk (or -1 if none)
    - 'response_offset_start': offset within response for local logits
    - 'response_offset_end': offset within response for local logits
    """
    cp_rank = parallel_state.cp_rank

    local_start = cp_rank * chunk_size
    local_end = (cp_rank + 1) * chunk_size

    results = []
    seq_start = 0

    for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
        seq_end = seq_start + total_length
        prompt_length = total_length - response_length

        # Global logits positions for this sequence
        logits_global_start = seq_start + prompt_length - 1
        logits_global_end = seq_start + total_length - 1

        local_logits_start = max(logits_global_start, local_start)
        local_logits_end = min(logits_global_end, local_end)

        if local_logits_start < local_logits_end:
            local_logits_start_rel = local_logits_start - local_start
            local_logits_end_rel = local_logits_end - local_start

            resp_offset_start = local_logits_start - logits_global_start
            resp_offset_end = local_logits_end - logits_global_start
        else:
            local_logits_start_rel = -1
            local_logits_end_rel = -1
            resp_offset_start = -1
            resp_offset_end = -1

        results.append(
            {
                "local_logits_start": local_logits_start_rel,
                "local_logits_end": local_logits_end_rel,
                "response_offset_start": resp_offset_start,
                "response_offset_end": resp_offset_end,
            }
        )

        seq_start = seq_end

    return results


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    parallel_state: ParallelState,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
):
    """
    All offsets start from the begining of the prompt.
    """
    cp_rank = parallel_state.cp_rank
    cp_size = parallel_state.cp_size
    assert cp_size > 1

    prompt_length = total_length - response_length
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    parallel_state: ParallelState,
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    chunk_size: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Calculate correct sample mean for CP
    """
    cp_size = parallel_state.cp_size
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    elif parallel_state.uses_contiguous_cp:
        if chunk_size is None:
            total_packed_len = sum(total_lengths)
            chunk_size = (total_packed_len + cp_size - 1) // cp_size

        offsets = get_packed_batch_offsets_with_allgather_cp(
            total_lengths, response_lengths, parallel_state, chunk_size
        )

        local_chunk_info = []
        for offset, loss_mask in zip(offsets, loss_masks, strict=False):
            if offset["local_logits_start"] >= 0:
                start = offset["response_offset_start"]
                end = offset["response_offset_end"]
                local_mask = loss_mask[start:end]
                local_len = offset["local_logits_end"] - offset["local_logits_start"]
            else:
                local_mask = torch.tensor([], device=loss_mask.device, dtype=loss_mask.dtype)
                local_len = 0

            local_chunk_info.append(
                {
                    "local_len": local_len,
                    "local_mask": local_mask,
                    "full_mask": loss_mask,
                }
            )

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            # Continue the gradient flow when the result is zero.
            total = 0.0 * x.sum()
            offset = 0
            for info in local_chunk_info:
                if info["local_len"] > 0:
                    x_chunk = x[offset : offset + info["local_len"]]
                    local_mask = info["local_mask"]
                    full_mask = info["full_mask"]
                    total = total + (x_chunk * local_mask).sum() / torch.clamp_min(full_mask.sum(), 1)
                    offset += info["local_len"]
            return total

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            total = 0.0 * x.sum()
            offset = 0
            for info in local_chunk_info:
                if info["local_len"] > 0:
                    x_chunk = x[offset : offset + info["local_len"]]
                    local_mask = info["local_mask"]
                    total = total + (x_chunk * local_mask).sum()
                    offset += info["local_len"]
            return total

    else:
        cp_chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, parallel_state, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
                    for x_i, chunked_loss_mask, loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, loss_masks, strict=False
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def all_gather_with_cp(
    tensor: torch.Tensor, total_length: int, response_length: int, parallel_state: ParallelState
) -> torch.Tensor:
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.
    """
    cp_group = parallel_state.cp_group
    cp_size = parallel_state.cp_size

    if cp_size == 1:
        return tensor

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length, parallel_state)

    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len: int) -> torch.Tensor:
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    parallel_state: ParallelState,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    cp_rank = parallel_state.cp_rank
    cp_size = parallel_state.cp_size

    if qkv_format == "bshd":
        assert max_seq_len is not None

    def pad_tokens(tokens, pad):
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    if cp_size == 1:
        if qkv_format == "bshd":
            pad = max_seq_len - tokens.size(0)
            tokens = pad_tokens(tokens, pad)
        return tokens

    token_len = len(tokens)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    # get 2 chunk for thd cp
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    parallel_state: ParallelState,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
) -> list[float] | torch.Tensor:
    assert len(log_prob) == response_length

    cp_size = parallel_state.cp_size

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, parallel_state, qkv_format, max_token_len
    )

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)


def slice_packed_log_probs(
    rollout_log_probs: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    parallel_state: ParallelState,
    chunk_size: int,
) -> list[torch.Tensor]:
    """
    Slice per-sequence log_probs based on packed batch layout for all-gather CP.
    """
    cp_size = parallel_state.cp_size
    cp_rank = parallel_state.cp_rank

    if cp_size == 1:
        return rollout_log_probs

    # Local chunk boundaries in packed space
    local_start = cp_rank * chunk_size
    local_end = (cp_rank + 1) * chunk_size

    sliced_log_probs = []
    seq_start = 0  # Start position of current sequence in packed space

    for total_length, response_length, log_prob in zip(
        total_lengths, response_lengths, rollout_log_probs, strict=False
    ):
        seq_end = seq_start + total_length
        prompt_length = total_length - response_length

        logits_global_start = seq_start + prompt_length - 1
        logits_global_end = seq_start + total_length - 1

        intersect_start = max(logits_global_start, local_start)
        intersect_end = min(logits_global_end, local_end)

        if intersect_start < intersect_end:
            lp_start = intersect_start - logits_global_start
            lp_end = intersect_end - logits_global_start
            sliced_log_probs.append(log_prob[lp_start:lp_end])
        else:
            sliced_log_probs.append(log_prob.new_empty(0))

        seq_start = seq_end

    return sliced_log_probs
