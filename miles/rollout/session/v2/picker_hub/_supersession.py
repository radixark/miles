"""The leaf-supersession trim shared by the session v2 pick hooks."""

import logging
from collections.abc import Callable

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def drop_superseded_leaves(
    leaf_samples: list[Sample], session_metadata: dict, *, sibling_prompt_must_match: bool
) -> list[Sample]:
    """Trim every leaf a later sibling supersedes; order the survivors.

    A later child of a leaf's parent supersedes that leaf; with
    ``sibling_prompt_must_match`` only a child with the leaf's exact prompt
    tokens does. Roots have no parent, so a later root supersedes a root leaf
    only when their prompt tokens match: roots with different prompts
    (subagents) all survive. Survivors are ordered by checkpoint count, then
    commit order, both descending.
    """
    tree = session_metadata["tree"]
    nodes = {n["id"]: n for n in tree["nodes"]}
    roots = [n["id"] for n in tree["nodes"] if n["parent"] is None]
    children: dict[int, list[int]] = {}
    for n in tree["nodes"]:
        if n["parent"] is not None:
            children.setdefault(n["parent"], []).append(n["id"])
    leaf_rows = tree["leaves"]
    same_prompt = _same_prompt_fn(leaf_samples, nodes)

    kept: list[Sample] = []
    for sample in leaf_samples:
        descriptor = sample.metadata["leaf"]
        leaf_id, parent = descriptor["node_id"], descriptor["parent"]
        later = [node for node in (roots if parent is None else children[parent]) if node > leaf_id]
        if parent is None or sibling_prompt_must_match:
            later = [node for node in later if same_prompt(node, leaf_id)]
        if not later:
            kept.append(sample)
            continue
        # Wall-clock regressions are diagnostic; `seq` is the ordering contract.
        clock_regressions = [
            (sibling, nodes[sibling]["committed_at"])
            for sibling in later
            if nodes[sibling]["committed_at"] < nodes[leaf_id]["committed_at"]
        ]
        if clock_regressions:
            logger.warning(
                "Picker detected wall-clock rollback for superseded leaf "
                "(response_id=%r, seq=%d, committed_at=%s); "
                "later siblings=%s; continuing by seq",
                descriptor["response_id"],
                leaf_id,
                nodes[leaf_id]["committed_at"],
                clock_regressions,
            )

        # Length is diagnostic only; temporal supersession is decided by `seq`.
        survivors_max = max(
            nodes[row["node_id"]]["num_tokens"]
            for row in leaf_rows
            if any(sibling in row["path_node_ids"] for sibling in later)
        )
        if nodes[leaf_id]["num_tokens"] > survivors_max:
            logger.warning(
                "Picker trimming superseded leaf (response_id=%r, seq=%d) "
                "even though it is longer than every later sibling's deepest leaf "
                "(%d > %d tokens); continuing by seq; use a custom picker to keep it",
                descriptor["response_id"],
                leaf_id,
                nodes[leaf_id]["num_tokens"],
                survivors_max,
            )
        logger.info("Picker trimmed superseded leaf seq=%d", leaf_id)
    return sorted(
        kept,
        key=lambda sample: (
            len(sample.metadata["leaf"]["path_node_ids"]),
            sample.metadata["leaf"]["node_id"],
        ),
        reverse=True,
    )


def _same_prompt_fn(leaf_samples: list[Sample], nodes: dict[int, dict]) -> Callable[[int, int], bool]:
    """Compare two nodes' prompt tokens, read from any leaf sample whose path covers each node."""
    covering_sample: dict[int, Sample] = {}
    for sample in leaf_samples:
        for node_id in sample.metadata["leaf"]["path_node_ids"]:
            covering_sample.setdefault(node_id, sample)

    def prompt(node_id: int) -> list[int] | None:
        sample = covering_sample.get(node_id)
        if sample is None:
            return None
        return sample.metadata["accumulated_token_ids"][: nodes[node_id]["completion_span"][0]]

    def same_prompt(node_id: int, other_id: int) -> bool:
        if nodes[node_id]["completion_span"][0] != nodes[other_id]["completion_span"][0]:
            return False
        node_prompt = prompt(node_id)
        return node_prompt is not None and node_prompt == prompt(other_id)

    return same_prompt
