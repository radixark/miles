"""Trie packing must train every sample exactly as if it were packed alone."""

import random

import pytest
import torch
from tests.fast.backends.training_utils.reference_trie_attention import reference_trie_attention

from miles.backends.training_utils.data.trie_packing import (
    TrieKV,
    build_trie_layout,
    pack_trie_bins,
    routing_aware_packing_keys,
    trie_replay_rows,
)
from miles.utils.token_trie_builder import TokenTrieBuilder


def _samples(seed: int) -> list[list[int]]:
    """GRPO-shaped samples: a shared system prompt, per-group prompts, then responses.

    Includes identical prompts, a sample that leaves its group's prompt early,
    and a later call that extends an earlier one, which are the cases a trie
    must keep apart or merge correctly.
    """
    rng = random.Random(seed)
    system = [rng.randrange(1, 50) for _ in range(23)]
    tokens = []
    for _ in range(3):
        prompt = system + [rng.randrange(1, 50) for _ in range(rng.randrange(5, 15))]
        for sample in range(4):
            own = prompt if sample else prompt[: len(system) + 2]
            if sample == 3:
                own = own + [rng.randrange(1, 50) for _ in range(4)]
            # a small alphabet, so responses often start alike and share their first tokens
            response = [rng.randrange(1, 4) for _ in range(rng.randrange(3, 9))]
            tokens.append(own + response)
    return tokens


def _layouts(tokens, width=10_000, window_left=None, pad_multiple=1, window_start_multiple=1):
    return [
        build_trie_layout(
            trie,
            window_left=window_left,
            window_start_multiple=window_start_multiple,
            pad_multiple=pad_multiple,
        )
        for trie in pack_trie_bins(tokens, width)
    ]


def _sample_token_indices(layout) -> dict[int, torch.Tensor]:
    """Each sample's token indices: the leading keys gathered for the segment holding its last token."""
    token_indices = {}
    for sample, segment, length in zip(
        layout.sample_indices, layout.sample_segments, layout.sample_lengths, strict=True
    ):
        start = int(layout.kv.cu_seqlens[segment])
        token_indices[sample] = layout.kv.index[start : start + length]
    return token_indices


def test_shared_tokens_are_stored_once():
    """The point of the layout: the 3 groups' common prompt is not repeated per sample."""
    tokens = _samples(0)
    (layout,) = _layouts(tokens)
    assert layout.num_tokens < sum(len(t) for t in tokens) // 2


def test_each_sample_reads_back_from_its_token_indices():
    """A sample's token indices hold its tokens at their logical positions, in every bin."""
    tokens = _samples(1)
    layouts = _layouts(tokens, width=60, pad_multiple=8)
    assert len(layouts) > 1
    for layout in layouts:
        for sample, token_indices in _sample_token_indices(layout).items():
            assert layout.tokens[token_indices].tolist() == tokens[sample]
            assert layout.positions[token_indices].tolist() == list(range(len(tokens[sample])))


def test_a_sample_longer_than_the_bin_trains_alone():
    """As under standard packing, a sample over the bin width gets a trie of its own."""
    tokens = _samples(6)
    width = max(len(t) for t in tokens) + 10
    oversized = tokens[0][:20] + list(range(1, 2 * width))
    bins = pack_trie_bins(tokens + [oversized], width)
    (alone,) = [trie for trie in bins if len(tokens) in trie.tags]
    assert alone.tags == (len(tokens),)
    assert len(alone.token_ids) == len(oversized)
    rest = [trie for trie in bins if trie is not alone]
    assert all(len(trie.token_ids) <= width for trie in rest)
    assert sorted(i for trie in rest for i in trie.tags) == list(range(len(tokens)))


def test_empty_samples_are_rejected():
    with pytest.raises(ValueError, match="at least one token"):
        pack_trie_bins([[1, 2], []], 8)


def test_identical_samples_share_every_token():
    """Two rollouts with the same tokens are stored once, and each reads all of it back."""
    tokens = [[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]
    (layout,) = _layouts(tokens)
    assert layout.num_tokens == 5
    token_indices = _sample_token_indices(layout)
    assert torch.equal(token_indices[0], token_indices[1])


def test_a_sample_ending_inside_another_shares_its_segment():
    """A sample the other continues ends inside the other's segment, and reads back only its own tokens."""
    tokens = [[5, 6, 7], [5, 6, 7, 8, 9]]
    (layout,) = _layouts(tokens)
    assert layout.cu_seqlens_q.tolist() == [0, 5]
    token_indices = _sample_token_indices(layout)
    assert token_indices[0].tolist() == [0, 1, 2]
    assert token_indices[1].tolist() == [0, 1, 2, 3, 4]


def test_unrelated_samples_get_roots_of_their_own():
    """Samples with no first token in common share nothing."""
    tokens = [[1, 2, 3, 4], [1, 2, 3, 5], [6, 7, 8, 9], [6, 7, 8, 10], [1, 2, 11], [6, 12]]
    (layout,) = _layouts(tokens)
    # roots (1, 2) and (6,), under which (3)->(4)|(5) and (11), and (7, 8)->(9)|(10) and (12)
    assert layout.positions[layout.cu_seqlens_q[:-1].long()].tolist().count(0) == 2
    assert layout.num_tokens == (2 + 1 + 1 + 1 + 1) + (1 + 2 + 1 + 1 + 1)
    for sample, token_indices in _sample_token_indices(layout).items():
        assert layout.tokens[token_indices].tolist() == tokens[sample]


def test_replay_rows_take_each_token_from_a_sample_that_holds_it():
    """A token replays the row of a sample that holds it, at the token's logical position; padding replays none."""
    tokens = _samples(7)
    rows = [torch.tensor([[1000 * sample + p] for p in range(len(t))]) for sample, t in enumerate(tokens)]
    for layout in _layouts(tokens, width=60, pad_multiple=8):
        holders = {}
        for sample, token_indices in _sample_token_indices(layout).items():
            for token_index in token_indices.tolist():
                holders.setdefault(token_index, set()).add(sample)
        for token_index, row in enumerate(trie_replay_rows(layout, rows)[:, 0].tolist()):
            sample, position = divmod(row, 1000)
            if token_index in holders:
                assert sample in holders[token_index] and position == layout.positions[token_index]
            else:
                assert row == -1


def test_routing_keys_share_a_token_only_where_the_recorded_routing_agrees():
    """Samples whose rollouts routed a shared token apart part there; each still reads back its own tokens."""
    tokens = [torch.tensor([5, 6, 7, 8, 9])] * 3
    agree = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])  # [T - 1, layers, top_k]
    apart = agree.clone()
    apart[2, 0, 1] = 9
    packing_keys = routing_aware_packing_keys(tokens, [agree, agree, apart])
    (trie,) = pack_trie_bins([t.tolist() for t in tokens], 100, packing_keys=packing_keys)
    layout = build_trie_layout(trie)
    assert layout.num_tokens == 5 + 3  # the third sample shares only the two tokens before the one routed apart
    for sample, token_indices in _sample_token_indices(layout).items():
        assert layout.tokens[token_indices].tolist() == tokens[sample].tolist()


def test_routing_keyed_rows_replay_each_sample_its_own_routing():
    """Packed on routing keys, every token a sample reads replays the sample's own recorded routing, even where
    rollouts routed a shared prompt token or response opening apart."""
    tokens = _samples(7)
    rows = []
    for sample, ids in enumerate(tokens):
        # a function of the prefix, as when samples hit one radix-cache entry, except where some samples part
        route = [[hash(tuple(ids[: p + 1])) % 1000] for p in range(len(ids) - 1)]
        if sample % 3 == 1:
            route[20][0] += 1000  # inside the system prompt every sample shares
        if sample % 4 == 2:
            route[-2][0] += 2000  # inside a response
        rows.append(torch.tensor(route))
    packing_keys = routing_aware_packing_keys([torch.tensor(ids) for ids in tokens], rows)
    padded = [torch.cat([r, r.new_full((1, 1), -1)]) for r in rows]
    for trie in pack_trie_bins(tokens, 60, packing_keys=packing_keys):
        layout = build_trie_layout(trie, pad_multiple=8)
        replayed = trie_replay_rows(layout, padded)
        for sample, token_indices in _sample_token_indices(layout).items():
            assert layout.tokens[token_indices].tolist() == tokens[sample]
            assert torch.equal(replayed[token_indices[:-1]], rows[sample])


def test_a_sample_ending_inside_a_segment_gives_its_rows_only_to_its_own_tokens():
    """The segment's tokens past the shorter sample's end replay the longer sample's rows, in either order."""
    for first, second in (((1, 2), (1, 2, 3)), ((1, 2, 3), (1, 2))):
        builder = TokenTrieBuilder()
        builder.add(first, tag=0)
        builder.add(second, tag=1)
        rows = [torch.full((len(first), 1), 0), torch.full((len(second), 1), 1)]
        replayed = trie_replay_rows(build_trie_layout(builder.flatten()), rows)
        assert replayed[2, 0] == (1 if len(first) < len(second) else 0)


def test_window_gather_keeps_only_reachable_ancestors():
    """A windowed layer needs at most `window_left` ancestor tokens per segment."""
    tokens = _samples(3)
    (full,) = _layouts(tokens)
    (windowed,) = _layouts(tokens, window_left=6)
    q_lengths = full.cu_seqlens_q.diff()
    ancestors = full.kv.cu_seqlens.diff() - q_lengths
    kept = windowed.window_kv.cu_seqlens.diff() - q_lengths
    assert torch.equal(kept, ancestors.clamp(max=6))
    assert windowed.window_kv.index.numel() < full.kv.index.numel()


def test_a_window_gather_starts_on_its_start_multiple():
    """Moving the start down to a multiple only prepends ancestor keys the window masks."""
    tokens = _samples(3)
    (windowed,) = _layouts(tokens, window_left=6)
    (rounded,) = _layouts(tokens, window_left=6, window_start_multiple=8)
    extended = 0
    for segment in range(len(windowed.cu_seqlens_q) - 1):
        lo, hi = windowed.window_kv.cu_seqlens[segment : segment + 2].tolist()
        a_lo, a_hi = rounded.window_kv.cu_seqlens[segment : segment + 2].tolist()
        window = windowed.window_kv.index[lo:hi]
        gathered = rounded.window_kv.index[a_lo:a_hi]
        extra = gathered.numel() - window.numel()
        assert 0 <= extra < 8 and torch.equal(gathered[extra:], window)
        positions = rounded.positions[gathered]
        first = int(positions[0])
        assert first % 8 == 0
        assert torch.equal(positions, torch.arange(first, first + gathered.numel()))
        extended += extra > 0
    assert extended


def _unpacked_attention(query, key, value, layout, window_left):
    """Per-sample causal attention over physical tensors gathered into each sample."""
    outputs = {}
    for sample, token_indices in _sample_token_indices(layout).items():
        full = torch.tensor([0, len(token_indices)], dtype=torch.int32)
        out = reference_trie_attention(
            query.index_select(0, token_indices),
            key,
            value,
            TrieKV(index=token_indices, cu_seqlens=full, max_seqlen=len(token_indices)),
            full,
            window_left=window_left,
        )
        outputs[sample] = (token_indices, out)
    return outputs


@pytest.mark.parametrize(("window_left", "window_start_multiple"), [(None, 1), (5, 1), (5, 4)])
def test_trie_attention_matches_unpacked_attention(window_left, window_start_multiple):
    """Outputs and gradients match packing each sample alone, with every token a sample reads, shared ones
    included, counted once for each sample that reads it."""
    torch.manual_seed(0)
    tokens = _samples(5)
    (layout,) = _layouts(tokens, window_left=window_left, pad_multiple=8, window_start_multiple=window_start_multiple)
    kv = layout.kv if window_left is None else layout.window_kv
    shape = (layout.num_tokens, 4, 8)
    query = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    key = torch.randn(shape[0], 2, 8, dtype=torch.float64, requires_grad=True)
    value = torch.randn(shape[0], 2, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(shape, dtype=torch.float64)

    packed = reference_trie_attention(query, key, value, kv, layout.cu_seqlens_q, window_left=window_left)
    alone = _unpacked_attention(query, key, value, layout, window_left)

    packed_loss, alone_loss = 0, 0
    for token_indices, out in alone.values():
        assert torch.allclose(packed[token_indices], out)
        packed_loss = packed_loss + (packed[token_indices] * weight[token_indices]).sum()
        alone_loss = alone_loss + (out * weight[token_indices]).sum()
    packed_grads = torch.autograd.grad(packed_loss, (query, key, value))
    alone_grads = torch.autograd.grad(alone_loss, (query, key, value))
    for packed_grad, alone_grad in zip(packed_grads, alone_grads, strict=True):
        assert torch.allclose(packed_grad, alone_grad)
