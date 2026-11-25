"""Build flat token tries (``miles.utils.flat_token_trie``) from token sequences."""

from __future__ import annotations

from dataclasses import dataclass

from miles.utils.flat_token_trie import FlatTrieSegment, IndexedFlatTrie


def _common_prefix_end(
    seq_a: tuple[int, ...],
    seq_b: tuple[int, ...],
    start: int,
    end: int,
) -> int:
    """First position in ``[start, end)`` where the sequences differ, else ``end``."""
    # Whole-slice equality is a single C-level tuple compare, an order of
    # magnitude cheaper per token than a Python loop, and sequences that share a
    # long prompt agree over the whole run far more often than they diverge.
    if seq_a[start:end] == seq_b[start:end]:
        return end
    for position in range(start, end):
        if seq_a[position] != seq_b[position]:
            return position
    return end


@dataclass(slots=True)
class _TrieNode:
    """The builder's mutable form of a ``FlatTrieSegment``.

    The run is ``tokens[start:end]``, a view into one of the inserted token sequences.
    ``start`` doubles as the segment's ``start_depth``, because a token's
    depth in the trie is its position, so cutting a run only moves two integers
    and never copies tokens. ``terminals`` holds ``(entry, end)`` for each
    sequence that ends in the run, at depth ``end``.
    """

    tokens: tuple[int, ...]
    start: int
    end: int
    children: dict[int, _TrieNode]
    terminals: list[tuple[int, int]]


class TokenTrieBuilder:
    """Incrementally build a token trie and emit a compact DFS layout.

    Token sequences must arrive as non-empty tuples of ints: ``trie_packing.pack_trie_bins``
    converts and checks its input once, because doing it on every call here would
    cost more than the walk.
    """

    def __init__(self) -> None:
        # A zero-length virtual root, so every walk has a node to extend and the
        # first insertion needs no special case. It is never emitted.
        self._root = _TrieNode(tokens=(), start=0, end=0, children={}, terminals=[])
        # The tag of each sequence added so far, duplicates included, in the order
        # they arrived. Nodes name a sequence by its entry here.
        self._tags: list[int] = []
        self._size = 0

    @property
    def size(self) -> int:
        """Tokens the trie stores: a token several sequences share counts once."""
        return self._size

    def _deepest_match(self, token_seq: tuple[int, ...]) -> tuple[int, _TrieNode]:
        """Deepest depth ``token_seq`` already occupies, and the node holding it.

        ``depth < node.end`` means the match ends inside that node's run, so a
        caller that wants to extend the node must cut its run there first.
        """
        node = self._root
        depth = 0
        while depth < len(token_seq):
            child = node.children.get(token_seq[depth])
            if child is None:
                break
            node = child
            depth = _common_prefix_end(child.tokens, token_seq, depth, min(child.end, len(token_seq)))
            if depth < child.end:
                break
        return depth, node

    def insertion_cost(self, token_seq: tuple[int, ...]) -> int:
        """How much ``add(token_seq)`` would grow ``size``, without adding it.

        That is the tokens of ``token_seq`` past the longest prefix of it the trie
        already stores: 0 for a sequence the trie holds whole (a duplicate, or one
        that ends inside a stored sequence), ``len(token_seq)`` for one that shares
        no first token with it.
        """
        depth, _ = self._deepest_match(token_seq)
        return len(token_seq) - depth

    def add(self, token_seq: tuple[int, ...], *, tag: int) -> None:
        """Insert ``token_seq`` with ``tag``, which the trie never reads: ``flatten`` hands it back beside the
        index of the sequence's last token."""
        token_seq_entry = len(self._tags)
        depth, node = self._deepest_match(token_seq)
        self._size += len(token_seq) - depth
        # A run is cut only where sequences fork, so every node boundary is a fork or
        # a leaf, and a sequence may end anywhere inside a run.
        if depth == len(token_seq):
            node.terminals.append((token_seq_entry, depth))
        elif depth == node.end and not node.children and node is not self._root:
            # The sequence continues a leaf: extend the leaf's run with it.
            node.tokens, node.end = token_seq, len(token_seq)
            node.terminals.append((token_seq_entry, len(token_seq)))
        else:
            if depth < node.end:
                # The sequence forks off inside the node's run: cut the run there. The
                # node keeps the part before the cut, which its parent points at, and
                # the sequences that end in it; a new node takes the rest, with the
                # node's children and the sequences that end after the cut.
                rest = _TrieNode(
                    tokens=node.tokens,
                    start=depth,
                    end=node.end,
                    children=node.children,
                    terminals=[terminal for terminal in node.terminals if terminal[1] > depth],
                )
                node.terminals = [terminal for terminal in node.terminals if terminal[1] <= depth]
                node.end, node.children = depth, {node.tokens[depth]: rest}
            node.children[token_seq[depth]] = _TrieNode(
                tokens=token_seq,
                start=depth,
                end=len(token_seq),
                children={},
                terminals=[(token_seq_entry, len(token_seq))],
            )
        self._tags.append(tag)

    def flatten(self) -> IndexedFlatTrie:
        """Flatten depth-first, recording each sequence's tag and where its last token sits.

        A sequence's last token locates all of it: the ancestors are recoverable from the
        layout, which is why nothing here accumulates them. Threading the full
        index list down the walk instead would cost one int per logical token
        — the one output a trie has no way to compress.
        """
        if not self._tags:
            raise ValueError("cannot flatten an empty trie")
        segments: list[FlatTrieSegment] = []
        token_ids: list[int] = []
        last_token_indices: list[int] = [-1] * len(self._tags)
        pending: list[tuple[_TrieNode, int | None]] = [
            (child, None) for child in reversed(self._root.children.values())
        ]
        while pending:
            node, parent_segment = pending.pop()
            token_ids.extend(node.tokens[node.start : node.end])
            segment_index = len(segments)
            segments.append(
                FlatTrieSegment(parent_idx=parent_segment, length=node.end - node.start, start_depth=node.start)
            )
            for token_seq_entry, end in node.terminals:
                last_token_indices[token_seq_entry] = len(token_ids) - 1 - (node.end - end)
            pending.extend((child, segment_index) for child in reversed(node.children.values()))
        return IndexedFlatTrie(
            token_ids=tuple(token_ids),
            segments=tuple(segments),
            tags=tuple(self._tags),
            last_token_indices=tuple(last_token_indices),
        )
