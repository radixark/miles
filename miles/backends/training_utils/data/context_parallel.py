import logging
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    cp_rank: int | None = None,
    cp_size: int | None = None,
):
    """
    All offsets start from the begining of the prompt.

    ``cp_rank`` / ``cp_size`` default to this process's parallel state; pass them
    explicitly to compute another rank's offsets outside the process group.
    """
    if cp_rank is None or cp_size is None:
        parallel_state = get_parallel_state()
        cp_rank = parallel_state.cp.rank
        cp_size = parallel_state.cp.size
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


class LocalResponseRows(NamedTuple):
    """Where one sample's response sits in this rank's flattened logits.

    ``row_ranges[i]`` are half-open local logit rows that score response positions
    ``response_ranges[i]``, one to one; logit row ``t`` scores token ``t + 1``.
    """

    row_ranges: tuple[tuple[int, int], ...]
    response_ranges: tuple[tuple[int, int], ...]

    def tokens(self, sample_tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        """The tokens these rows score."""
        pieces = [sample_tokens[prompt_length + start : prompt_length + end] for start, end in self.response_ranges]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def response_indices(self) -> list[int]:
        """The response position each row scores."""
        return [index for start, end in self.response_ranges for index in range(start, end)]


def zigzag_response_ranges(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    *,
    cp_rank: int | None = None,
    cp_size: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The response positions a zigzag CP rank holds, one range per zigzag half; an empty half is ``(0, 0)``."""
    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
    )
    prompt_length = total_length - response_length
    first, second = (
        (start - prompt_length, end - prompt_length) if start < end else (0, 0) for start, end in tokens_offset
    )
    return first, second


def iter_local_response_rows(
    num_rows: int,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    allgather_cp: bool,
    max_seq_lens: list[int] | None = None,
) -> Iterator[LocalResponseRows]:
    """Per sample, the rows of this rank's ``[num_rows, V]`` logits that score its response.

    Without context parallelism a response is one run of rows. Under all-gather CP this rank holds
    at most one run of it; under zigzag CP it holds two, one per zigzag half, and either may be empty.
    """
    cp = get_parallel_state().cp
    slot_start = 0  # first local row of the sample's slot, without CP or under zigzag CP
    seq_start = 0  # first global token of the sample, under all-gather CP
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        if cp.size == 1:
            if qkv_format == "bshd":
                slot_start = max_seq_len * i
            end = slot_start + total_length - 1
            sample_rows = LocalResponseRows(((end - response_length, end),), ((0, response_length),))
            slot_start += total_length
        elif allgather_cp:
            # thd splits the concatenated samples into one contiguous piece per rank; bshd splits
            # each padded sample on its own, so every sample has a piece of max_seq_len // cp rows
            piece_rows = max_seq_len // cp.size if qkv_format == "bshd" else num_rows
            sample_rows = _response_rows_allgather_cp(
                local_start=piece_rows * i if qkv_format == "bshd" else 0,
                global_start=cp.rank * piece_rows,
                piece_rows=piece_rows,
                seq_start=0 if qkv_format == "bshd" else seq_start,
                total_length=total_length,
                response_length=response_length,
            )
        else:
            chunk_size, chunks_offset, _, _ = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            response_ranges = zigzag_response_ranges(total_length, response_length, qkv_format, max_seq_len)
            prompt_length = total_length - response_length
            row_ranges = tuple(
                _zigzag_half_rows(
                    slot_start + half * chunk_size, response_ranges[half], chunk_start - prompt_length + 1
                )
                for half, (chunk_start, _) in enumerate(chunks_offset)
            )
            sample_rows = LocalResponseRows(row_ranges, response_ranges)
            slot_start += 2 * chunk_size
        seq_start += total_length

        assert all(
            0 <= start <= end <= num_rows for start, end in sample_rows.row_ranges
        ), f"{sample_rows.row_ranges} vs {num_rows} rows"
        assert all(
            row_end - row_start == response_end - response_start
            for (row_start, row_end), (response_start, response_end) in zip(
                sample_rows.row_ranges, sample_rows.response_ranges, strict=True
            )
        ), f"{sample_rows}"
        yield sample_rows


def _response_rows_allgather_cp(
    *, local_start, global_start, piece_rows, seq_start, total_length, response_length
) -> LocalResponseRows:
    """At most one run: the response's rows within this rank's contiguous piece of the sequence.

    The piece is global rows ``[global_start, global_start + piece_rows)``, held from local row
    ``local_start``; the sample starts at global row ``seq_start``.
    """
    logit_start = seq_start + total_length - response_length - 1
    logit_end = seq_start + total_length - 1
    start, end = max(logit_start, global_start), min(logit_end, global_start + piece_rows)
    if end <= start:
        return LocalResponseRows(((local_start, local_start),), ((0, 0),))
    offset = local_start - global_start
    return LocalResponseRows(((start + offset, end + offset),), ((start - logit_start, end - logit_start),))


def _zigzag_half_rows(half_start: int, response_range: tuple[int, int], first_position: int) -> tuple[int, int]:
    """Local rows of one zigzag half whose first row scores response position ``first_position``.

    A half holding none of the response stays empty at its own start.
    """
    start, end = response_range
    if start >= end:
        return half_start, half_start
    return half_start + start - first_position, half_start + end - first_position


def slice_loss_masks_for_local_cp(
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Backward-compatible wrapper for local CP response mask slicing."""
    return get_local_response_loss_masks(total_lengths, response_lengths, loss_masks, qkv_format, max_seq_lens)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    *,
    denominators: list[torch.Tensor] | torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Calculate correct sample mean for CP; ``denominators`` overrides each
    sample's own ``loss_mask.sum()`` (e.g. pass ``rollout_mask_sums`` for
    per-rollout means)."""
    if denominators is None:
        denominators = [m.sum() for m in loss_masks]

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(denominator, 1)
                    for x_i, loss_mask_i, denominator in zip(
                        x.split(response_lengths, dim=0), loss_masks, denominators, strict=True
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=True)
                ]
            )

    else:
        cp_chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=True)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunked_loss_masks.append(
                slice_log_prob_with_cp(loss_mask, total_length, response_length, qkv_format, max_seq_len)
            )
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(denominator, 1)
                    for x_i, chunked_loss_mask, denominator in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, denominators, strict=True
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=True
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def get_local_response_loss_masks(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Return response loss masks aligned with this rank's local log-probs."""
    parallel_state = get_parallel_state()
    if parallel_state.cp.size == 1:
        return loss_masks

    local_masks = []
    for i, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=True)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        local_masks.append(slice_log_prob_with_cp(loss_mask, total_length, response_length, qkv_format, max_seq_len))

    return local_masks


def all_gather_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """This rank's zigzag share of a per-response tensor, gathered over the CP group to the full response.

    Each rank scatters its share into zeros and the all-reduce sums the shares. The result needs a
    gradient exactly when ``tensor`` does, which is the same on every CP rank (log-probs on all of
    them, data on none), so every rank runs the same backward all-reduce; a rank holding none of
    the response still stays in the graph through its empty share.
    """
    parallel_state = get_parallel_state()
    if parallel_state.cp.size == 1:
        return tensor
    ranges = zigzag_response_ranges(total_length, response_length, qkv_format, max_seq_len)
    positions = torch.cat([torch.arange(start, end, device=tensor.device) for start, end in ranges])
    assert positions.numel() == tensor.size(0), f"{positions.numel()} positions vs {tensor.size(0)} values"
    full_tensor = tensor.new_zeros((response_length, *tensor.shape[1:])).index_copy(0, positions, tensor)
    return dist.nn.all_reduce(full_tensor, group=parallel_state.cp.group)


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    parallel_state: object | None = None,
) -> torch.Tensor:
    """
    Slice tokens into the local zigzag CP layout.
    """
    if parallel_state is None:
        parallel_state = get_parallel_state()
    cp_rank = parallel_state.cp.rank
    cp_size = parallel_state.cp.size

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


def natural_to_zigzag_slice(tensor: torch.Tensor, dim: int, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Slice a full-length tensor into the zigzag ring-attention CP layout.

    Rank ``cp_rank`` owns chunks ``[cp_rank, 2*cp_size - 1 - cp_rank]`` from the
    ``2*cp_size`` equal-sized partitions along ``dim``. This is the inverse of
    an all-gather over the zigzag CP layout (hence "natural → zigzag").

    Unlike :func:`slice_with_cp`, this helper does not pad — it expects the
    input to already be divisible by ``2 * cp_size`` along ``dim``. If not, it
    prints a warning and returns the tensor unchanged.
    """
    total = tensor.shape[dim]
    num_chunks = 2 * cp_size
    if total % num_chunks != 0:
        print(f"Warning: dim {dim} size {total} not divisible by 2*cp_size={num_chunks}")
        return tensor

    chunk_size = total // num_chunks
    chunk_indices = [cp_rank, 2 * cp_size - 1 - cp_rank]

    slices = [tensor.narrow(dim, idx * chunk_size, chunk_size) for idx in chunk_indices]
    return torch.cat(slices, dim=dim)


def allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits: torch.Tensor,
    args,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits: Model output, used only for its local row count. For ``bshd``,
            each batch row has its own sequence window.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = get_parallel_state().cp.group

    # thd logits are [1, T_local, V] and bshd [B, S_local, V]; either way flattened rows
    num_rows = logits.size(0) * logits.size(1)
    layout = list(
        iter_local_response_rows(
            num_rows,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            qkv_format=args.qkv_format,
            allgather_cp=True,
            max_seq_lens=max_seq_lens,
        )
    )

    for key, values in res.items():
        # each rank pads its contiguous share to the full response; the shares do not overlap
        full_resps = []
        for value, sample_rows, response_length in zip(values, layout, response_lengths, strict=True):
            ((start, end),) = sample_rows.response_ranges
            full_resp = F.pad(value, (0, 0) * (value.ndim - 1) + (start, response_length - end))
            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
) -> list[float] | torch.Tensor:
    """This rank's zigzag share of any per-response-token list or tensor (log-probs, masks, values)."""
    assert len(log_prob) == response_length
    if get_parallel_state().cp.size == 1:
        return log_prob
    (start_0, end_0), (start_1, end_1) = zigzag_response_ranges(
        total_length, response_length, qkv_format, max_token_len
    )
    if isinstance(log_prob, list):
        return log_prob[start_0:end_0] + log_prob[start_1:end_1]
    return torch.cat([log_prob[start_0:end_0], log_prob[start_1:end_1]], dim=0)


def assemble_log_prob_from_cp(
    chunks: dict[int, torch.Tensor],
    total_length: int,
    response_length: int,
    cp_size: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Inverse of `slice_log_prob_with_cp`: per-rank slices back to one response.

    `chunks` maps cp_rank to that rank's slice; every rank must be present.
    Offsets come from the same helper the forward split uses.
    """
    assert cp_size > 1, "no reassembly needed at cp_size=1"
    missing = sorted(set(range(cp_size)) - set(chunks))
    assert not missing, f"cp ranks {missing} missing; cannot reassemble a partial group"

    out = torch.zeros(response_length, dtype=next(iter(chunks.values())).dtype)
    for cp_rank, chunk in chunks.items():
        ranges = zigzag_response_ranges(
            total_length, response_length, qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
        )
        taken = 0
        for start, end in ranges:
            out[start:end] = chunk[taken : taken + end - start]
            taken += end - start
        assert taken == len(chunk), f"cp rank {cp_rank}: consumed {taken} of {len(chunk)} values"
    return out
