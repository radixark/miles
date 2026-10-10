import torch
import triton
import triton.language as tl


_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}
SCORE_ROW_ALIGN = 4
_SELECT_BLOCK = 512
_SELECT_RADIX_BITS = 4
# Sorting a verified proposal beats a compaction pass over the row at topk 512 and loses at 2048 (H200, 16k).
_SELECT_SORT_MAX_TOPK = 1024


def torch_dsa_topk(logits: torch.Tensor, topk: int, row_starts=None, row_ends=None) -> torch.Tensor:
    score, indices = torch.topk(logits, topk, dim=-1)
    indices = indices.to(torch.int32)
    return indices.masked_fill(score == -torch.inf, -1)


def flashinfer_dsa_topk(logits: torch.Tensor, topk: int, row_starts=None, row_ends=None) -> torch.Tensor:
    import flashinfer
    from sglang.srt.environ import envs

    orig_shape = logits.shape
    # flashinfer reads dense rows; indexer scores carry SCORE_ROW_ALIGN padding
    logits = logits.reshape(-1, logits.shape[-1]).contiguous()

    score, indices = flashinfer.top_k(
        logits,
        topk,
        sorted=False,
        deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
        tie_break=flashinfer_tie_break_value(),
        dsa_graph_safe=True,
    )
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(score == -torch.inf, -1)
    if len(orig_shape) > 2:
        indices = indices.reshape(*orig_shape[:-1], topk)
    return indices


@triton.jit
def _ordered_key(x):
    bits = tl.where(x == 0.0, 0.0, x).to(tl.uint32, bitcast=True)
    flip = tl.where(
        (bits >> 31) != 0, tl.full(bits.shape, 0xFFFFFFFF, tl.uint32), tl.full(bits.shape, 0x80000000, tl.uint32)
    )
    return bits ^ flip


@triton.jit
def _radix_threshold(row_ptr, start, end, block_lo, BLOCK: tl.constexpr, BITS: tl.constexpr, TOPK: tl.constexpr):
    NUM_BINS: tl.constexpr = 1 << BITS
    bins = tl.arange(0, NUM_BINS)
    threshold = tl.full([], 0, tl.uint32)
    remaining = tl.full([], TOPK, tl.int32)
    take_whole_bin = remaining < 0
    for radix_pass in tl.static_range(32 // BITS):
        shift = 32 - BITS * (radix_pass + 1)
        if take_whole_bin == 0:
            hist = tl.zeros([NUM_BINS], dtype=tl.int32)
            for off in range(block_lo, end, BLOCK):
                cols = off + tl.arange(0, BLOCK)
                in_range = (cols >= start) & (cols < end)
                x = tl.load(row_ptr + cols, mask=in_range, other=-float("inf"))
                key = _ordered_key(x)
                match = in_range & (x != -float("inf"))
                if radix_pass > 0:
                    match = match & ((key >> (shift + BITS)) == (threshold >> (shift + BITS)))
                hist += tl.histogram(((key >> shift) & (NUM_BINS - 1)).to(tl.int32), NUM_BINS, mask=match)
            total = tl.sum(hist)
            at_or_above = total - tl.cumsum(hist, 0) + hist
            bin_id = tl.maximum(tl.sum((at_or_above >= remaining).to(tl.int32)) - 1, 0)
            above = tl.sum(tl.where(bins == bin_id, at_or_above - hist, 0))
            in_bin = tl.sum(tl.where(bins == bin_id, hist, 0))
            take_all = total <= remaining
            take_whole_bin = take_all | (in_bin == remaining - above)
            threshold = tl.where(take_all, threshold, threshold | (bin_id.to(tl.uint32) << shift))
            remaining = tl.where(take_all, remaining, remaining - above)
    return threshold, remaining, take_whole_bin


@triton.jit
def _select_topk_kernel(
    logits_ptr,
    stride_logits,
    row_starts_ptr,
    row_ends_ptr,
    hint_ptr,
    stride_hint,
    out_ptr,
    stride_out,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    BITS: tl.constexpr,
    SORT_PROPOSAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(row_starts_ptr + row)
    end = tl.load(row_ends_ptr + row)
    row_ptr = logits_ptr + row * stride_logits
    out_row = out_ptr + row * stride_out
    block_lo = start // BLOCK * BLOCK
    slots = tl.arange(0, TOPK_PAD)

    threshold = tl.full([], 0, tl.uint32)
    remaining = tl.full([], TOPK, tl.int32)
    select_all = (end - start) <= TOPK
    threshold_exact = select_all
    hint_is_answer = tl.full([], 0, tl.int1)
    if select_all == 0:
        hint = tl.load(hint_ptr + row * stride_hint + slots, mask=slots < TOPK, other=-1)
        hint_in_range = (hint >= start) & (hint < end)
        hint_x = tl.load(row_ptr + hint, mask=hint_in_range, other=float("inf"))
        hint_key = tl.where(hint_in_range, _ordered_key(hint_x), tl.full([TOPK_PAD], 0xFFFFFFFF, tl.uint32))
        threshold = tl.min(hint_key, 0)
        count_above = tl.zeros([BLOCK], tl.int32)
        count_equal = tl.zeros([BLOCK], tl.int32)
        count_valid = tl.zeros([BLOCK], tl.int32)
        for off in range(block_lo, end, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            in_range = (cols >= start) & (cols < end)
            x = tl.load(row_ptr + cols, mask=in_range, other=-float("inf"))
            valid = in_range & (x != -float("inf"))
            key = _ordered_key(x)
            count_above += (valid & (key > threshold)).to(tl.int32)
            count_equal += (valid & (key == threshold)).to(tl.int32)
            count_valid += valid.to(tl.int32)
        n_above = tl.sum(count_above)
        n_equal = tl.sum(count_equal)
        select_all = tl.sum(count_valid) <= TOPK
        threshold_exact = select_all | ((n_above < TOPK) & (n_above + n_equal >= TOPK))
        remaining = TOPK - n_above
        if SORT_PROPOSAL & (select_all == 0) & (n_above + n_equal == TOPK):
            hint_valid = hint_in_range & (hint_x != -float("inf"))
            sorted_hint = tl.sort(tl.where(hint_valid, hint, 2147483647), 0)
            tl.store(out_row + slots, sorted_hint, mask=slots < TOPK)
            tl.debug_barrier()
            next_hint = tl.load(out_row + slots + 1, mask=slots < TOPK - 1, other=-1)
            no_duplicate = tl.sum((next_hint == sorted_hint).to(tl.int32)) == 0
            hint_is_answer = (tl.sum(hint_valid.to(tl.int32)) == TOPK) & no_duplicate
    take_whole_bin = select_all
    if threshold_exact == 0:
        threshold, remaining, take_whole_bin = _radix_threshold(row_ptr, start, end, block_lo, BLOCK, BITS, TOPK)
    if select_all:
        threshold = tl.full([], 0, tl.uint32)

    n_out = tl.full([], 0, tl.int32)
    n_equal_seen = tl.full([], 0, tl.int32)
    for off in range(block_lo, tl.where(hint_is_answer, block_lo, end), BLOCK):
        cols = off + tl.arange(0, BLOCK)
        in_range = (cols >= start) & (cols < end)
        x = tl.load(row_ptr + cols, mask=in_range, other=-float("inf"))
        valid = in_range & (x != -float("inf"))
        key = _ordered_key(x)
        above = valid & ((key > threshold) | (take_whole_bin & (key == threshold)))
        equal = (valid & (key == threshold) & (take_whole_bin == 0)).to(tl.int32)
        equal_rank = tl.cumsum(equal, 0) - equal + n_equal_seen
        selected = (above | ((equal != 0) & (equal_rank < remaining))).to(tl.int32)
        tl.store(out_row + tl.cumsum(selected, 0) - selected + n_out, cols, mask=selected != 0)
        n_out += tl.sum(selected)
        n_equal_seen += tl.sum(equal)
    if hint_is_answer == 0:
        tl.store(out_row + slots, tl.full([TOPK_PAD], -1, tl.int32), mask=(slots >= n_out) & (slots < TOPK))


def select_topk(
    logits: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    proposal: torch.Tensor,
    num_warps: int = 4,
) -> torch.Tensor:
    """Exact top-k of logits[i, row_starts[i]:row_ends[i]]: int32 column ids, ascending, -1 padded; ties at the
    k-th value keep the smaller column and -inf is never picked. proposal [rows, topk] (-1 allowed) only sets the
    speed: a row whose proposal fails the one-pass check runs an in-kernel radix select. Columns outside a row's
    range are never read."""
    rows = logits.shape[0]
    out = torch.empty(rows, topk, dtype=torch.int32, device=logits.device)
    if rows == 0:
        return out
    _select_topk_kernel[(rows,)](
        logits,
        logits.stride(0),
        row_starts,
        row_ends,
        proposal,
        proposal.stride(0),
        out,
        out.stride(0),
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        BLOCK=_SELECT_BLOCK,
        BITS=_SELECT_RADIX_BITS,
        SORT_PROPOSAL=topk <= _SELECT_SORT_MAX_TOPK,
        num_warps=num_warps,
    )
    return out


def canonical_dsa_topk(
    logits: torch.Tensor, topk: int, row_starts: torch.Tensor, row_ends: torch.Tensor
) -> torch.Tensor:
    """select_topk with the rollout's radix top-k as the proposal. Score rows must be SCORE_ROW_ALIGN-aligned;
    the proposal kernel overwrites up to three columns just left of each row's range."""
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    assert logits.stride(-1) == 1 and logits.stride(0) % SCORE_ROW_ALIGN == 0, "pad score rows to SCORE_ROW_ALIGN"
    row_starts = row_starts.to(torch.int32)
    row_ends = row_ends.to(torch.int32)
    proposal = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=logits.device)
    if logits.shape[0] > 0:
        topk_transform_ragged_v2(
            logits,
            (row_ends - row_starts).clamp(min=0),
            out_offsets=row_starts,
            out_indices=proposal,
            row_starts=row_starts,
        )
    return select_topk(logits, topk, row_starts, row_ends, proposal)


def get_dsa_topk_fn(topk_backend: str):
    if topk_backend == "torch":
        return torch_dsa_topk
    if topk_backend == "flashinfer":
        return flashinfer_dsa_topk
    if topk_backend == "canonical":
        return canonical_dsa_topk
    raise ValueError(f"Unsupported miles DSA topk backend: {topk_backend}")


def flashinfer_tie_break_value() -> int:
    from sglang.srt.environ import envs

    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
