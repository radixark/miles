"""A prefix trie must lay out every sequence exactly, storing each shared prefix once."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from miles.utils.flat_token_trie import FlatTrie, FlatTrieSegment, IndexedFlatTrie
from miles.utils.token_trie_builder import TokenTrieBuilder, _common_prefix_end


def _token_parents_and_depths(trie: FlatTrie) -> tuple[list[int], list[int]]:
    """Each token's parent token index (``-1`` on a root) and depth, read off the segments."""
    parents: list[int] = []
    depths: list[int] = []
    segment_ends: list[int] = []
    for segment in trie.segments:
        start = len(parents)
        first_parent = -1 if segment.parent_idx is None else segment_ends[segment.parent_idx] - 1
        parents.extend([first_parent, *range(start, start + segment.length - 1)])
        depths.extend(range(segment.start_depth, segment.start_depth + segment.length))
        segment_ends.append(start + segment.length)
    return parents, depths


def _token_indices_per_seq(packed: IndexedFlatTrie) -> tuple[tuple[int, ...], ...]:
    """The token indices each sequence visits, rebuilt from its last token's index.

    The trie stores only the index of a sequence's last token, on the claim that
    it locates the whole sequence against the parent array. Every assertion about the layout below runs
    through this, so that claim is under test rather than assumed.
    """
    parents = _token_parents_and_depths(packed)[0]
    all_token_indices: list[tuple[int, ...]] = []
    for last in packed.last_token_indices:
        token_indices, token_index = [], last
        while token_index >= 0:
            token_indices.append(token_index)
            token_index = parents[token_index]
        all_token_indices.append(tuple(reversed(token_indices)))
    return tuple(all_token_indices)


def _pack_one(token_seqs: tuple[tuple[int, ...], ...]) -> IndexedFlatTrie:
    builder = TokenTrieBuilder()
    for index, token_seq in enumerate(token_seqs):
        builder.add(token_seq, tag=index)
    return builder.flatten()


def _assert_layout_matches_token_seqs(token_seqs: tuple[tuple[int, ...], ...]) -> None:
    """Every sequence reads back from its last token's index, each token sits at its depth on every sequence
    through it, the trie holds each distinct prefix once, and its segments are maximal: none has exactly one child."""
    packed = _pack_one(token_seqs)
    depths = _token_parents_and_depths(packed)[1]
    for source_index, token_indices in zip(packed.tags, _token_indices_per_seq(packed), strict=True):
        assert token_seqs[source_index] == tuple(packed.token_ids[token_index] for token_index in token_indices)
        assert [depths[token_index] for token_index in token_indices] == list(range(len(token_indices)))
    prefixes = {token_seq[:end] for token_seq in token_seqs for end in range(1, len(token_seq) + 1)}
    assert len(packed.token_ids) == len(prefixes)
    children = Counter(segment.parent_idx for segment in packed.segments)
    assert all(children[index] != 1 for index in range(len(packed.segments)))


def test_arbitrary_depth_trie_layout():
    token_seqs = ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6))

    packed = _pack_one(token_seqs)
    token_indices_by_tag = dict(zip(packed.tags, _token_indices_per_seq(packed), strict=True))

    assert packed.token_ids == (1, 2, 3, 4, 5, 6)
    assert tuple(token_indices_by_tag[index] for index in range(len(token_seqs))) == (
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 5),
    )
    assert _token_parents_and_depths(packed) == ([-1, 0, 1, 2, 2, 1], [0, 1, 2, 3, 3, 2])


def test_branching_mid_run_splits_the_shared_prefix():
    # The second sequence diverges inside the first sequence's run, so the run has
    # to be broken in two rather than the shared tokens re-emitted.
    token_seqs = ((1, 2, 3, 4, 5), (1, 2, 8, 9))

    packed = _pack_one(token_seqs)

    assert packed.token_ids == (1, 2, 3, 4, 5, 8, 9)
    assert _token_indices_per_seq(packed) == ((0, 1, 2, 3, 4), (0, 1, 5, 6))
    _assert_layout_matches_token_seqs(token_seqs)


def test_duplicate_and_prefix_sequences_add_no_tokens():
    token_seqs = ((1, 2, 3), (1, 2, 3), (1, 2))

    packed = _pack_one(token_seqs)

    assert packed.token_ids == (1, 2, 3)
    assert _token_indices_per_seq(packed) == ((0, 1, 2), (0, 1, 2), (0, 1))
    _assert_layout_matches_token_seqs(token_seqs)


def test_a_sequence_ending_inside_another_leaves_its_run_whole():
    # A run is cut only where sequences fork, whichever of the two arrives first.
    for token_seqs in (((1, 2), (1, 2, 3)), ((1, 2, 3), (1, 2))):
        packed = _pack_one(token_seqs)

        assert packed.segments == (FlatTrieSegment(parent_idx=None, length=3, start_depth=0),)
        _assert_layout_matches_token_seqs(token_seqs)


def test_a_fork_splits_the_sequences_ending_in_the_run_it_cuts():
    # (1, 2) ends inside the run (1, 2, 3); a later fork cuts that run before (1, 4) or right after (1, 2, 4) its end.
    for fork, expected in (
        ((1, 4), (FlatTrieSegment(None, 1, 0), FlatTrieSegment(0, 2, 1), FlatTrieSegment(0, 1, 1))),
        ((1, 2, 4), (FlatTrieSegment(None, 2, 0), FlatTrieSegment(0, 1, 2), FlatTrieSegment(0, 1, 2))),
    ):
        token_seqs = ((1, 2, 3), (1, 2), fork)
        packed = _pack_one(token_seqs)

        assert packed.segments == expected
        _assert_layout_matches_token_seqs(token_seqs)


def test_insertion_cost_counts_only_the_tokens_a_sequence_adds():
    # Bin packing budgets on these numbers, so they have to stay exact.
    builder = TokenTrieBuilder()
    builder.add((1, 2, 3, 4), tag=0)

    assert builder.insertion_cost((1, 2, 9, 9)) == 2
    assert builder.insertion_cost((1, 2, 3, 4)) == 0
    assert builder.insertion_cost((1, 2)) == 0
    assert builder.insertion_cost((1, 2, 3, 4, 5, 6, 7)) == 3
    assert builder.insertion_cost((8, 9)) == 2
    assert builder.size == 4

    builder.add((1, 2, 9, 9), tag=1)
    assert builder.size == 6
    assert len(builder.flatten().token_ids) == 6


def test_unrelated_sequences_form_independent_roots():
    token_seqs = ((1, 2, 3), (4, 5), (1, 2, 6))

    packed = _pack_one(token_seqs)
    token_indices_by_tag = dict(zip(packed.tags, _token_indices_per_seq(packed), strict=True))

    assert packed.token_ids == (1, 2, 3, 6, 4, 5)
    assert tuple(token_indices_by_tag[index] for index in range(len(token_seqs))) == ((0, 1, 2), (4, 5), (0, 1, 3))
    _assert_layout_matches_token_seqs(token_seqs)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_layout_matches_independent_sequences(seed):
    rng = random.Random(seed)
    next_token = 1
    token_seqs: list[tuple[int, ...]] = []
    for _ in range(12):
        if token_seqs and rng.random() < 0.8:
            source = rng.choice(token_seqs)
            token_seq = list(source[: rng.randrange(len(source) + 1)])
        else:
            token_seq = []
        extension_length = rng.randint(1, 8)
        token_seq.extend(range(next_token, next_token + extension_length))
        next_token += extension_length
        token_seqs.append(tuple(token_seq))
    token_seqs.extend((token_seqs[0], token_seqs[1][:-1] or token_seqs[1]))

    _assert_layout_matches_token_seqs(tuple(token_seqs))


def test_common_prefix_end_finds_the_first_difference():
    rng = random.Random(2)
    for _ in range(500):
        seq_a = tuple(rng.randrange(2) for _ in range(rng.randint(1, 40)))
        seq_b = tuple(token if rng.random() < 0.95 else 1 - token for token in seq_a)
        start = rng.randint(0, len(seq_a) - 1)
        end = rng.randint(start, len(seq_a))

        expected = next((p for p in range(start, end) if seq_a[p] != seq_b[p]), end)

        assert _common_prefix_end(seq_a, seq_b, start, end) == expected
