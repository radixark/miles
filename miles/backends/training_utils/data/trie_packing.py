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
