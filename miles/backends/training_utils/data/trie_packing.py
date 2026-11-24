"""Shared-prefix (trie) packing for training micro-batches.

Samples of one step share long prompt prefixes: a GRPO group's rollouts start
from one prompt, and different prompts reuse system prompts. Packing a
micro-batch's samples into one prefix trie stores every shared token once. Each
trie segment then attends to its ancestors and itself, which a stock varlen
kernel computes once the segment's keys and values are gathered as
``[ancestors; segment]`` and the causal mask is aligned bottom-right.

Samples merge wherever their tokens agree, responses included: tokens that
agree on everything before them have the same hidden states. Each sample reads
its rows back through its own token indices, a row several samples share once for each,
so every per-sample, per-token value stays the sample's own.
"""

from __future__ import annotations

import bisect
import functools
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from miles.utils.flat_token_trie import IndexedFlatTrie
from miles.utils.token_trie_builder import TokenTrieBuilder


def _best_fit_trie_index(
    bin_builders: Sequence[TokenTrieBuilder],
    token_seq: tuple[int, ...],
    max_bin_width: int,
) -> int | None:
    """The bin with the least room left after adding ``token_seq``, or ``None`` if it fits in none."""
    best: tuple[int, int] | None = None
    for bin_index, builder in enumerate(bin_builders):
        remaining = max_bin_width - builder.size - builder.insertion_cost(token_seq)
        if remaining < 0:
            continue
        candidate = (remaining, bin_index)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[1]


def pack_trie_bins(
    tokens: Sequence[Sequence[int]],
    max_bin_width: int,
) -> list[IndexedFlatTrie]:
    """Best-fit samples into tries of at most ``max_bin_width`` tokens before padding.

    A sample longer than ``max_bin_width`` gets a trie of its own, as it trains alone under standard packing: no
    other sample fits beside it.
    """
    if max_bin_width <= 0:
        raise ValueError(f"max_bin_width must be positive, got {max_bin_width}")
    token_seqs = [tuple(sample) for sample in tokens]
    if not all(token_seqs):
        raise ValueError("samples must contain at least one token")

    bin_builders: list[TokenTrieBuilder] = []
    # Best-fit decreasing: place the long samples while the bins are still empty
    # and let short ones fill the gaps. Sorting on raw length rather than
    # insertion cost is deliberate, since that cost depends on what a bin
    # already holds, which is what this loop is still deciding.
    for sample_index in sorted(range(len(token_seqs)), key=lambda index: len(token_seqs[index]), reverse=True):
        token_seq = token_seqs[sample_index]
        bin_index = _best_fit_trie_index(bin_builders, token_seq, max_bin_width)
        if bin_index is None:
            bin_index = len(bin_builders)
            bin_builders.append(TokenTrieBuilder())
        bin_builders[bin_index].add(token_seq, tag=sample_index)
    return [builder.flatten() for builder in bin_builders]


@dataclass(frozen=True)
class TrieKV:
    """Which physical tokens each segment's queries attend to, segment by segment."""

    index: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


@dataclass(frozen=True)
class TrieLayout:
    """One packed trie row: physical tokens plus everything attention and loss need.

    Entry ``c`` of ``tokens`` and ``positions`` is the id and logical position of
    token index ``c``. Tokens run in depth-first order, and trailing padding forms
    its own segment.
    Samples are named by their index in the pool handed to ``pack_trie_bins``.
    ``sample_segments[i]`` is the segment holding sample ``sample_indices[i]``'s
    last token and ``sample_lengths[i]`` the sample's length: the first
    ``sample_lengths[i]`` entries of that segment's ``kv`` gather are the
    sample's token indices, position by position.
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    kv: TrieKV
    window_kv: TrieKV | None
    sample_indices: tuple[int, ...]
    sample_segments: tuple[int, ...]
    sample_lengths: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return self.tokens.numel()


def _ranges_to_index(ranges: list[tuple[int, int]]) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long)
    return torch.cat([torch.arange(start, end) for start, end in ranges])


def _gathered_kv(
    *,
    ancestors: list[list[int]],
    physical_starts: list[int],
    start_positions: list[int],
    lengths: list[int],
    window_left: int | None,
    start_multiple: int = 1,
) -> TrieKV:
    """Each segment's keys: its ancestors' tokens, then its own.

    With ``window_left``, only the last ``window_left`` ancestor tokens, extended back so that the keys
    start at a logical position that is a multiple of ``start_multiple``.
    """
    ranges: list[tuple[int, int]] = []
    seqlens: list[int] = []
    for segment, chain in enumerate(ancestors):
        ancestor_tokens = start_positions[segment]
        if window_left is not None:
            ancestor_tokens = min(ancestor_tokens, window_left)
            ancestor_tokens += (start_positions[segment] - ancestor_tokens) % start_multiple
        # Walk the chain from the parent upwards, taking whole or partial
        # segments until ``ancestor_tokens`` tokens are covered.
        taken: list[tuple[int, int]] = []
        remaining = ancestor_tokens
        for ancestor in reversed(chain):
            if remaining == 0:
                break
            take = min(remaining, lengths[ancestor])
            end = physical_starts[ancestor] + lengths[ancestor]
            taken.append((end - take, end))
            remaining -= take
        ranges.extend(reversed(taken))
        ranges.append((physical_starts[segment], physical_starts[segment] + lengths[segment]))
        seqlens.append(ancestor_tokens + lengths[segment])
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)
    return TrieKV(index=_ranges_to_index(ranges), cu_seqlens=cu_seqlens, max_seqlen=max(seqlens, default=0))


# Sliding layers attend over a window gather: only the ancestor keys the window reaches, far fewer than all of
# them under a long shared prefix. FA2 tiles each sequence's keys from its start (64 or 128 keys, by head dim),
# so starting each window gather on a logical position that is a multiple of 128 cuts the window into the same
# tiles as for an unpacked sample, and trie and standard packing round alike. FA3 and FA4 tile differently;
# there the extra keys are only masked out.
WINDOW_START_MULTIPLE = 128


def build_trie_layout(
    trie: IndexedFlatTrie,
    *,
    window_left: int | None = None,
    window_start_multiple: int = 1,
    pad_multiple: int = 1,
    pad_token_id: int = 0,
) -> TrieLayout:
    """Lay out one packed trie for a micro-batch.

    ``window_left`` is the sliding window's left extent (the kernel's
    ``window_size[0]``); with it the layout also carries the gather a windowed
    layer needs, which keeps only the ancestor tokens the window can reach.
    ``window_start_multiple`` moves the start of that gather down to a logical position
    that is a multiple of it, so a kernel that tiles keys from the start of each
    sequence cuts the window into the same tiles as for an unpacked sample.
    """
    segments = trie.segments
    lengths = [segment.length for segment in segments]
    start_positions = [segment.start_depth for segment in segments]
    physical_starts = [0] * len(segments)
    for index in range(1, len(segments)):
        physical_starts[index] = physical_starts[index - 1] + lengths[index - 1]
    ancestors: list[list[int]] = []
    for segment in segments:
        ancestors.append([] if segment.parent_idx is None else ancestors[segment.parent_idx] + [segment.parent_idx])
    # the segment holding each sample's last token, which may sit anywhere in it, and the sample's length
    sample_segments = [bisect.bisect_right(physical_starts, last) - 1 for last in trie.last_token_indices]
    sample_lengths = [
        start_positions[segment] + last - physical_starts[segment] + 1
        for segment, last in zip(sample_segments, trie.last_token_indices, strict=True)
    ]

    unpadded_width = len(trie.token_ids)
    padding = (-unpadded_width) % pad_multiple
    if padding:
        lengths = lengths + [padding]
        start_positions = start_positions + [0]
        physical_starts = physical_starts + [unpadded_width]
        ancestors = ancestors + [[]]
    tokens = torch.tensor(list(trie.token_ids) + [pad_token_id] * padding, dtype=torch.long)
    positions = torch.cat(
        [torch.arange(start, start + length) for start, length in zip(start_positions, lengths, strict=True)]
    )
    cu_seqlens_q = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    gathered_kv = functools.partial(
        _gathered_kv,
        ancestors=ancestors,
        physical_starts=physical_starts,
        start_positions=start_positions,
        lengths=lengths,
    )
    return TrieLayout(
        tokens=tokens,
        positions=positions,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(lengths),
        kv=gathered_kv(window_left=None),
        window_kv=(
            None if window_left is None else gathered_kv(window_left=window_left, start_multiple=window_start_multiple)
        ),
        sample_indices=tuple(trie.tags),
        sample_segments=tuple(sample_segments),
        sample_lengths=tuple(sample_lengths),
    )


def micro_batch_trie_layout(
    tokens: Sequence[Sequence[int]],
    total_lengths: Sequence[int],
    response_lengths: Sequence[int],
    model_spec: TrieModelSpec,
    *,
    window_start_multiple: int,
    pad_multiple: int,
) -> TrieLayout:
    """The one trie row holding a scheduled micro-batch's samples, laid out for ``model_spec``'s layers."""
    # a response's first token is predicted from the last prompt token's row
    if any(total - response < 1 for total, response in zip(total_lengths, response_lengths, strict=True)):
        raise ValueError("trie packing needs at least one prompt token per sample")
    bins = pack_trie_bins(tokens, sum(total_lengths))
    if len(bins) != 1:
        raise AssertionError(f"a micro-batch packed into {len(bins)} tries, expected 1")
    return build_trie_layout(
        bins[0],
        window_left=_left_extent(model_spec.sliding_window_size),
        window_start_multiple=window_start_multiple,
        pad_multiple=pad_multiple,
    )


def _left_extent(size: int | None) -> int | None:
    """How many earlier tokens a window of ``size`` tokens reaches: the kernels' left extent."""
    return None if size is None else size - 1


def _sample_token_indices(layout: TrieLayout) -> list[torch.Tensor]:
    """Each sample's token indices, position by position, in micro-batch (pool) order."""
    bounds = layout.kv.cu_seqlens.tolist()
    by_sample = {
        sample: layout.kv.index[bounds[segment] : bounds[segment] + length]
        for sample, segment, length in zip(
            layout.sample_indices, layout.sample_segments, layout.sample_lengths, strict=True
        )
    }
    return [by_sample[sample] for sample in range(len(by_sample))]


def response_logit_rows(
    layout: TrieLayout, total_lengths: Sequence[int], response_lengths: Sequence[int]
) -> torch.Tensor:
    """The rows of a trie row that predict each response's tokens, responses back to back.

    Row ``p`` of a sample predicts its token ``p + 1``, so a response of ``R`` tokens ending a sample of ``T``
    tokens is predicted by the sample's rows at positions ``T - R - 1 .. T - 2``. A row several samples share
    appears once for each, so each sample's loss reads its own copy.
    """
    rows = [
        token_indices[total - response - 1 : total - 1]
        for token_indices, total, response in zip(
            _sample_token_indices(layout), total_lengths, response_lengths, strict=True
        )
    ]
    return torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)


def trie_input_loss_mask(
    layout: TrieLayout,
    loss_masks: Sequence[torch.Tensor],
    total_lengths: Sequence[int],
    response_lengths: Sequence[int],
) -> torch.Tensor:
    """``[1, T]`` mask over a trie row: 1 at each token some sample's response trains."""
    mask = torch.zeros(1, layout.num_tokens, dtype=torch.int)
    for token_indices, loss_mask, total, response in zip(
        _sample_token_indices(layout), loss_masks, total_lengths, response_lengths, strict=True
    ):
        mask[0, token_indices[total - response : total]] |= loss_mask.to(device=mask.device, dtype=torch.int)
    return mask


def trie_replay_rows(layout: TrieLayout, rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """A trie row's per-token replay rows, from each sample's per-token rows in micro-batch (pool) order.

    ``rows[i]`` holds one row per token of sample ``i`` (``-1`` rows where the rollout recorded none, as for
    its final token). A token takes the row of a sample that holds it, at the token's logical
    position; padding tokens take ``-1`` rows. A token several samples share replays that one sample's
    rollout routing, so it is exact when the rollout routed those samples' shared prefix alike, as it does
    when they hit one radix-cache entry.
    """
    missing = rows[0].new_full((1, *rows[0].shape[1:]), -1)
    offsets = torch.tensor([0] + [r.size(0) for r in rows]).cumsum(0)
    flat = torch.cat([*rows, missing])
    index = torch.full((layout.num_tokens,), flat.size(0) - 1)
    # each sample's token indices lead the keys gathered for its segment
    for sample, segment, length in zip(
        layout.sample_indices, layout.sample_segments, layout.sample_lengths, strict=True
    ):
        start = int(layout.kv.cu_seqlens[segment])
        index[layout.kv.index[start : start + length]] = offsets[sample] + torch.arange(length)
    return flat.index_select(0, index.to(flat.device))


@dataclass(frozen=True)
class TrieModelSpec:
    """What a model's trie rows depend on, read from its config.

    ``sliding_window_size`` is the number of tokens a sliding layer attends over, if any layer slides.
    """

    sliding_window_size: int | None = None
