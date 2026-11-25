"""Prefix tries of token sequences, stored flat and compressed for custom attention layouts.

Every token sequence starts at position 0, never a suffix or a window. That is
what makes a token's depth in the trie equal its position in every sequence
running through it, which in turn is what lets the physical layout carry logical
positions. ``miles.utils.token_trie_builder`` builds these tries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlatTrie:
    """A token trie stored flat and compressed.

    Flat, in the sense of Abseil's ``flat_hash_map`` as against its
    ``node_hash_map``: two arrays instead of nodes linked by pointers.
    ``token_ids`` holds every stored token once, segment after segment in
    depth-first order, so a segment's descendants all follow it before its
    next sibling does. ``segments`` describes those runs, each naming its
    parent by index.

    Compressed: each maximal unbranched path is a single ``FlatTrieSegment``,
    so the tree's shape takes one entry per segment rather than one per token.
    """

    token_ids: tuple[int, ...]
    segments: tuple[FlatTrieSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("flat trie requires at least one segment")
        segment_tokens = sum(segment.length for segment in self.segments)
        if len(self.token_ids) != segment_tokens:
            raise ValueError(f"flat trie has {len(self.token_ids)} token ids for segments of {segment_tokens} tokens")
        active_ancestors: list[int] = []
        for index, segment in enumerate(self.segments):
            if segment.length <= 0:
                raise ValueError(f"trie segment {index} has non-positive length {segment.length}")
            if segment.parent_idx is None:
                expected_start = 0
                active_ancestors = []
            else:
                if segment.parent_idx >= index or segment.parent_idx < 0:
                    raise ValueError(f"trie segment {index} has invalid parent {segment.parent_idx}")
                if segment.parent_idx not in active_ancestors:
                    raise ValueError(
                        "trie segments must be in depth-first order: "
                        f"parent {segment.parent_idx} is not active for segment {index}"
                    )
                parent = self.segments[segment.parent_idx]
                expected_start = parent.start_depth + parent.length
                active_ancestors = active_ancestors[: active_ancestors.index(segment.parent_idx) + 1]
            if segment.start_depth != expected_start:
                raise ValueError(
                    f"trie segment {index} starts at depth {segment.start_depth}, expected {expected_start}"
                )
            active_ancestors.append(index)


@dataclass(frozen=True)
class FlatTrieSegment:
    """A maximal unbranched path in the token trie, which a radix tree stores as
    one node.

    No sequence forks off before the segment's last token, so a sequence
    through its first token takes all of it or ends inside it. For the
    sequences (1, 2, 3, 4) and (1, 2, 5), the trie stores the tokens
    (1, 2, 3, 4, 5) as three segments:

        (1, 2)  FlatTrieSegment(parent_idx=None, length=2, start_depth=0)
        (3, 4)  FlatTrieSegment(parent_idx=0, length=2, start_depth=2)
        (5,)    FlatTrieSegment(parent_idx=0, length=1, start_depth=2)

    Adding (1, 2, 3) changes none of them: it ends inside (3, 4).
    """

    # Index of the parent segment in ``FlatTrie.segments``; None for a root.
    parent_idx: int | None
    # Number of tokens in the segment.
    length: int
    # Depth of the segment's first token in the trie, which is its position in
    # every sequence through it; not its index in ``FlatTrie.token_ids``.
    start_depth: int


@dataclass(frozen=True)
class IndexedFlatTrie(FlatTrie):
    """A capacity-bounded trie that remembers the token sequences it was built from.

    The two fields are parallel: entry ``i`` of each describes the same
    sequence. Read them as pairs and never index either one positionally, because
    their order is the packer's longest-first scratch order rather than the
    caller's. Nothing downstream depends on that order, only on the pairing.
    """

    # Each sequence's tag, as given to ``TokenTrieBuilder.add``: ``pack_trie_bins``
    # tags a sample with its position in the list it was handed. Bins partition
    # the sequences, so a bin has to record which ones it holds.
    tags: tuple[int, ...]
    # Where each sequence's last token sits in ``token_ids``. A sequence is a
    # walk down from a root, so this locates all of it through the segments'
    # parents without storing an index per token.
    last_token_indices: tuple[int, ...]
