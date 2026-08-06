"""The default pick hook: trim retry-superseded leaves."""

import logging

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def drop_retries(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Drop the dead leaves that retries leave behind.

    When a generation goes bad, the agent re-sends the same message and the
    conversation continues on the new branch — but the abandoned attempt is
    still a leaf in the tree, and it must not become a training sample.

    The rule, leaf by leaf: a sibling branched off the same parent after it
    -> this leaf is the abandoned attempt, superseded by the retry: trim it.
    No later sibling -> keep (root leaves included). Survivors are ordered by
    checkpoint count, then commit order, both descending.

    Example — turn 2 hit the length cap and the agent re-sent it (``seq`` is
    commit order):

        seq=0  turn 1: user asks, assistant answers
        ├── seq=1  turn 2 attempt: cut off at the length cap  (leaf) -> trim
        └── seq=2  turn 2 retry: the same user message re-sent
            └── seq=3  turn 3: the retry path continues       (leaf) -> keep
    """
    nodes = {n["id"]: n for n in session_metadata["tree"]["nodes"]}
    children: dict[int, list[int]] = {}
    for n in session_metadata["tree"]["nodes"]:
        if n["parent"] is not None:
            children.setdefault(n["parent"], []).append(n["id"])
    leaf_rows = session_metadata["tree"]["leaves"]

    kept: list[Sample] = []
    for sample in leaf_samples:
        descriptor = sample.metadata["leaf"]
        leaf_id, parent = descriptor["node_id"], descriptor["parent"]
        later = [] if parent is None else [sibling for sibling in children[parent] if sibling > leaf_id]
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
                "Default picker detected wall-clock rollback for superseded leaf "
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
                "Default picker trimming superseded leaf (response_id=%r, seq=%d) "
                "even though it is longer than every later sibling's deepest leaf "
                "(%d > %d tokens); continuing by seq; use a custom picker to keep it",
                descriptor["response_id"],
                leaf_id,
                nodes[leaf_id]["num_tokens"],
                survivors_max,
            )
        logger.info("Default picker trimmed superseded retry leaf seq=%d", leaf_id)
    return sorted(
        kept,
        key=lambda sample: (
            len(sample.metadata["leaf"]["path_node_ids"]),
            sample.metadata["leaf"]["node_id"],
        ),
        reverse=True,
    )
