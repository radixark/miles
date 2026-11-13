"""CPU radix-tree planning, independent of model and attention implementation."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixTreePlan:
    """Compact DFS rows plus original-occurrence and ancestor-path mappings.

    Sharing keys identify adapters/weight versions. Computation keys identify
    per-token replay or stochastic inputs; a mismatch forks the entire suffix.
    Targets and loss weights are deliberately absent from computation identity.
    """

    tokens: tuple[int, ...]
    positions: tuple[int, ...]
    sample_nodes: tuple[tuple[int, ...], ...]
    segment_offsets: tuple[int, ...]
    segment_parents: tuple[int, ...]
    kv_indices: tuple[int, ...]
    kv_offsets: tuple[int, ...]

    @property
    def expanded_tokens(self) -> int:
        return sum(map(len, self.sample_nodes))

    @property
    def max_path_length(self) -> int:
        return max(self.positions) + 1


def build_prefix_tree(
    sequences: Sequence[Sequence[int]],
    *,
    sharing_keys: Sequence[Hashable] | None = None,
    computation_keys: Sequence[Sequence[Hashable]] | None = None,
    share_prefixes: bool = True,
) -> PrefixTreePlan:
    """Build in O(expanded tokens + gathered KV rows), without GPU synchronization.

    ``share_prefixes=False`` produces the expanded, padding-free baseline using
    the same execution path. A sequence ending inside another ends a segment.
    Callers must include all computation-affecting side inputs in the keys.
    """
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("prefix training requires nonempty sequences")
    if sharing_keys is not None and len(sharing_keys) != len(sequences):
        raise ValueError("sharing_keys must have one entry per sequence")
    if computation_keys is not None and (
        len(computation_keys) != len(sequences)
        or any(len(keys) != len(sequence) for keys, sequence in zip(computation_keys, sequences, strict=True))
    ):
        raise ValueError("computation_keys must have one entry per input token")
    children, token_ids, depths, endpoints, paths = [{}], [0], [-1], set(), []
    for sample, sequence in enumerate(sequences):
        identity = sharing_keys[sample] if sharing_keys is not None else None
        if not share_prefixes:
            identity = (sample, identity)
        parent, path = 0, []
        for position, token in enumerate(sequence):
            replay = computation_keys[sample][position] if computation_keys is not None else None
            key = (identity, int(token), replay)
            child = children[parent].get(key)
            if child is None:
                child = len(children)
                children[parent][key] = child
                children.append({})
                token_ids.append(int(token))
                depths.append(position)
            path.append(child)
            parent = child
        endpoints.add(parent)
        paths.append(path)
    return _flatten(children, token_ids, depths, endpoints, paths)


def _flatten(children, token_ids, depths, endpoints, paths) -> PrefixTreePlan:
    tokens, positions, offsets, parents, kv_indices, kv_offsets = [], [], [0], [], [], [0]
    packed, segment_paths = {}, []
    stack = [(child, -1) for child in reversed(list(children[0].values()))]
    while stack:
        node, parent_segment = stack.pop()
        segment = len(parents)
        start = len(tokens)
        while True:
            packed[node] = len(tokens)
            tokens.append(token_ids[node])
            positions.append(depths[node])
            if node in endpoints or len(children[node]) != 1:
                break
            node = next(iter(children[node].values()))
        offsets.append(len(tokens))
        parents.append(parent_segment)
        ancestor_path = segment_paths[parent_segment] if parent_segment >= 0 else ()
        segment_path = (*ancestor_path, *range(start, len(tokens)))
        segment_paths.append(segment_path)
        kv_indices.extend(segment_path)
        kv_offsets.append(len(kv_indices))
        stack.extend((child, segment) for child in reversed(list(children[node].values())))
    return PrefixTreePlan(
        tokens=tuple(tokens),
        positions=tuple(positions),
        sample_nodes=tuple(tuple(packed[node] for node in path) for path in paths),
        segment_offsets=tuple(offsets),
        segment_parents=tuple(parents),
        kv_indices=tuple(kv_indices),
        kv_offsets=tuple(kv_offsets),
    )
