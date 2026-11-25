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

Every placeholder token of an image (or video) has the same id, so media tokens
merge on their item's content instead, and each item a trie row holds is one
run of tokens, fed to the model once.

With rollout routing replay, a shared token also replays one routing, so samples
share a token only where their rollouts routed it alike as well.
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from miles.utils.flat_token_trie import IndexedFlatTrie
from miles.utils.token_trie_builder import TokenTrieBuilder


# Above every vocabulary, so a media content id never equals a token id.
_MEDIA_ID_BASE = 1 << 32


@dataclass(frozen=True)
class TrieModality:
    """One media modality of a model's processor output (Qwen-VL style).

    Items of the modality expand into ``token_id`` placeholders; their patch rows and per-item ``(t, h, w)``
    grids sit under ``patches_key`` and ``grid_key``. An item has ``t * h * w`` patch rows and
    ``t * h * w / merge**2`` placeholder tokens.
    """

    token_id: int
    patches_key: str
    grid_key: str
    merge: int


@dataclass(frozen=True)
class MediaItem:
    """One image or video of a sample: its placeholder positions (ascending), patch rows and grid."""

    modality: int
    positions: torch.Tensor
    patches: torch.Tensor
    grid: torch.Tensor

    @functools.cached_property
    def content_id(self) -> int:
        """An id equal for two items exactly when their modality, grid and patches are, and unequal to every
        token id; hashed once, as the scheduler and the layout both read it."""
        digest = hashlib.blake2b(digest_size=8)
        digest.update(bytes([self.modality]))
        for tensor in (self.grid, self.patches):
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return _MEDIA_ID_BASE + int.from_bytes(digest.digest(), "little")


def media_items(
    tokens: torch.Tensor, multimodal: dict[str, torch.Tensor] | None, modalities: Sequence[TrieModality]
) -> list[MediaItem]:
    """A sample's media items, from its processor output: each item's placeholders, in order."""
    if not multimodal:
        return []
    known = {key for modality in modalities for key in (modality.patches_key, modality.grid_key)}
    if unknown := sorted(set(multimodal) - known):
        raise ValueError(f"--trie-packing does not lay out multimodal inputs {unknown}")
    items = []
    for index, modality in enumerate(modalities):
        if modality.grid_key not in multimodal:
            continue
        grids = multimodal[modality.grid_key]
        sizes = grids.prod(dim=-1).tolist()
        counts = [size // modality.merge**2 for size in sizes]
        positions = (tokens == modality.token_id).nonzero().squeeze(1).cpu()
        if positions.numel() != sum(counts):
            raise ValueError(
                f"{positions.numel()} placeholder tokens {modality.token_id}, but {modality.grid_key} expands to "
                f"{sum(counts)}"
            )
        patches = multimodal[modality.patches_key].split(sizes)
        for item_patches, grid, item_positions in zip(patches, grids, positions.split(counts), strict=True):
            items.append(MediaItem(index, item_positions, item_patches, grid))
    return items


def _media_token_seq(tokens: Sequence[int], media: Sequence[MediaItem]) -> list[int]:
    """A sample's token sequence for packing with media: each item's placeholders take its ``content_id``, so
    items merge only on equal content."""
    token_seq = list(tokens)
    for item in media:
        for position in item.positions.tolist():
            token_seq[position] = item.content_id
    return token_seq


def routing_aware_packing_keys(
    tokens: Sequence[torch.Tensor],
    routed_experts: Sequence[torch.Tensor],
    media: Sequence[Sequence[MediaItem]] | None = None,
) -> list[list[int]]:
    """Each sample's packing key per token under rollout routing replay.

    A shared token replays one routing, but the rollout can route it differently for two samples (prefilled on
    different engines, or decoded in different batches). A token's key is its id (its item's content id for a
    media placeholder) together with the routing the rollout recorded for it, so samples share a token only where
    both agree.

    ``tokens[i]`` holds sample ``i``'s token ids (lengths vary by sample), ``routed_experts[i]`` the experts the
    rollout routed each of them but the last to, ``[len(tokens[i]) - 1, layers, top_k]``, and ``media[i]`` its
    media items, if any.
    """
    packing_keys = []
    for index, (sample, experts) in enumerate(zip(tokens, routed_experts, strict=True)):
        ids = sample.tolist() if media is None else _media_token_seq(sample.tolist(), media[index])
        rows = [row.tobytes() for row in experts.flatten(1).numpy()] + [b""]
        packing_keys.append(
            [
                int.from_bytes(hashlib.blake2b(token.to_bytes(16, "little") + row, digest_size=8).digest(), "little")
                for token, row in zip(ids, rows, strict=True)
            ]
        )
    return packing_keys


def _pack_trie_bins_on_packing_keys(
    tokens: Sequence[Sequence[int]], max_bin_width: int, packing_keys: Sequence[Sequence[int]]
) -> list[IndexedFlatTrie]:
    """``pack_trie_bins`` on each sample's packing keys, with the token ids put back in the tries it returns."""
    token_ids = {}
    for sample, sample_packing_keys in zip(tokens, packing_keys, strict=True):
        token_ids.update(zip(sample_packing_keys, sample, strict=True))
    tries = pack_trie_bins(packing_keys, max_bin_width)
    return [dataclasses.replace(trie, token_ids=tuple(token_ids[key] for key in trie.token_ids)) for trie in tries]


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
    media: Sequence[Sequence[MediaItem]] | None = None,
    packing_keys: Sequence[Sequence[int]] | None = None,
) -> list[IndexedFlatTrie]:
    """Best-fit samples into tries of at most ``max_bin_width`` tokens before padding.

    A sample longer than ``max_bin_width`` gets a trie of its own, as it trains alone under standard packing: no
    other sample fits beside it. Samples merge where their ``packing_keys`` agree (``routing_aware_packing_keys``),
    else where their tokens do, media placeholders on their items' content.
    """
    if packing_keys is None and media is not None:
        packing_keys = [_media_token_seq(sample, items) for sample, items in zip(tokens, media, strict=True)]
    if packing_keys is not None:
        return _pack_trie_bins_on_packing_keys(tokens, max_bin_width, packing_keys)
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
    ``segment_parents[s]`` is the segment whose last token segment ``s``
    continues, ``-1`` for a root (padding included).
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    kv: TrieKV
    window_kv: TrieKV | None
    conv_kv: TrieKV | None
    conv_window_left: int | None
    sample_indices: tuple[int, ...]
    sample_segments: tuple[int, ...]
    sample_lengths: tuple[int, ...]
    segment_parents: tuple[int, ...]

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
    conv_window_left: int | None = None,
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
    ``conv_window_left`` is a causal convolution's left extent (kernel size - 1);
    with it the layout carries the same gather for that window, unaligned.
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
    parents = [-1 if segment.parent_idx is None else segment.parent_idx for segment in segments]

    unpadded_width = len(trie.token_ids)
    padding = (-unpadded_width) % pad_multiple
    if padding:
        lengths = lengths + [padding]
        start_positions = start_positions + [0]
        physical_starts = physical_starts + [unpadded_width]
        ancestors = ancestors + [[]]
        parents = parents + [-1]
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
        conv_kv=None if conv_window_left is None else gathered_kv(window_left=conv_window_left),
        conv_window_left=conv_window_left,
        sample_indices=tuple(trie.tags),
        sample_segments=tuple(sample_segments),
        sample_lengths=tuple(sample_lengths),
        segment_parents=tuple(parents),
    )


def micro_batch_trie_layout(
    tokens: Sequence[Sequence[int]],
    total_lengths: Sequence[int],
    response_lengths: Sequence[int],
    model_spec: TrieModelSpec,
    *,
    window_start_multiple: int,
    pad_multiple: int,
    media: Sequence[Sequence[MediaItem]] | None = None,
    packing_keys: Sequence[Sequence[int]] | None = None,
) -> TrieLayout:
    """The one trie row holding a scheduled micro-batch's samples, laid out for ``model_spec``'s layers."""
    # a response's first token is predicted from the last prompt token's row
    if any(total - response < 1 for total, response in zip(total_lengths, response_lengths, strict=True)):
        raise ValueError("trie packing needs at least one prompt token per sample")
    bins = pack_trie_bins(tokens, sum(total_lengths), media, packing_keys)
    if len(bins) != 1:
        raise AssertionError(f"a micro-batch packed into {len(bins)} tries, expected 1")
    return build_trie_layout(
        bins[0],
        window_left=_left_extent(model_spec.sliding_window_size),
        window_start_multiple=window_start_multiple,
        conv_window_left=_left_extent(model_spec.conv_kernel_size),
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


def trie_media_inputs(
    layout: TrieLayout, media: Sequence[Sequence[MediaItem]], modalities: Sequence[TrieModality]
) -> dict[str, torch.Tensor] | None:
    """The micro-batch's media as its trie row holds them: each item once, in token order.

    A model places an item's features at its placeholders in row order, so the items' tokens must not
    interleave, and every placeholder of the row must be one of theirs.
    """
    occurrences: dict[int, tuple[MediaItem, torch.Tensor]] = {}
    for token_indices, items in zip(_sample_token_indices(layout), media, strict=True):
        for item in items:
            item_token_indices = token_indices[item.positions]
            occurrences.setdefault(int(item_token_indices[0]), (item, item_token_indices))
    if not occurrences:
        return None
    ordered = [occurrences[first] for first in sorted(occurrences)]
    token_indices = torch.cat([item_token_indices for _, item_token_indices in ordered])
    if not bool((token_indices.diff() > 0).all()):
        raise AssertionError("media items' tokens interleave in the trie row")
    inputs = {}
    for index, modality in enumerate(modalities):
        items = [item for item, _ in ordered if item.modality == index]
        placeholders = int((layout.tokens == modality.token_id).sum())
        if placeholders != sum(item.positions.numel() for item in items):
            raise AssertionError(
                f"the trie row holds {placeholders} {modality.patches_key} placeholders, its items fewer"
            )
        if items:
            inputs[modality.patches_key] = torch.cat([item.patches for item in items])
            inputs[modality.grid_key] = torch.stack([item.grid for item in items])
    return inputs


def trie_replay_rows(layout: TrieLayout, rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """A trie row's per-token replay rows, from each sample's per-token rows in micro-batch (pool) order.

    ``rows[i]`` holds one row per token of sample ``i`` (``-1`` rows where the rollout recorded none, as for
    its final token). A token takes the row of a sample that holds it, at the token's logical
    position; padding tokens take ``-1`` rows. A token several samples share replays that one sample's
    rollout routing, which is each sharing sample's own when the row was packed on ``routing_aware_packing_keys``.
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
    ``conv_kernel_size`` is the kernel size of a causal convolution the model runs over the row, if any.
    ``media`` are the media modalities of the model's processor output.
    """

    sliding_window_size: int | None = None
    conv_kernel_size: int | None = None
    media: tuple[TrieModality, ...] = ()
