"""Per-leaf assembly and the pick/merge hook layer over the trajectory tree.

The policy layer of ``collect_samples`` (MULTI_LINEAGE_DESIGN v5), running
inside the session server process:

1. ``build_leaf_material`` (non-policy): one folded raw sample per leaf via
   the existing compute -> truncate -> fold primitives, each carrying its
   leaf descriptor and TITO bookkeeping in ``sample.metadata``.
2. A **pick hook** selects which leaf samples survive. The default is the
   temporal-supersession retry trim (ruling F).
3. A **merge hook** finalizes training samples: exactly-once completion
   masking across the surviving set, semantic-layer application, per-branch
   rewards keyed by response id.

Both hooks share the signature ``fn(leaf_samples, session_metadata) ->
list[Sample]`` and consume only those public inputs — the defaults dogfood
the same interface custom hooks get.
"""

import logging
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.merge import compute_samples_from_openai_records, truncate_samples_by_total_tokens
from miles.rollout.session.v2.session_state import SessionState
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_LEAF_KEY = "leaf"


def tree_metadata(state: SessionState) -> dict:
    """The structural layer: node and leaf tables, index-aligned with commits.

    ``response_id`` is the branch<->leaf join key — the agent saw the same id
    in each chat response, so the semantic layer can key per-branch data on it.
    """
    nodes = [
        {
            "id": node.seq,
            "parent": node.parent.seq if node.parent is not None else None,
            "seq": node.seq,
            "truncated": node.truncated,
            "committed_at": node.committed_at,
            "completion_span": list(node.completion_span),
            "num_tokens": len(node.token_ids),
            "response_id": node.response_id,
        }
        for node in state.tree.nodes
    ]
    leaves = [
        {"node_id": leaf.seq, "path_node_ids": [n.seq for n in leaf.path_nodes()]} for leaf in state.tree.leaves()
    ]
    return {"nodes": nodes, "leaves": leaves}


def build_leaf_material(
    args: Namespace,
    state: SessionState,
    tokenizer,
    *,
    max_seq_len: int | None,
    max_trim_tokens: int,
    compute_mismatch: Callable[[list[dict[str, Any]], list[int], Any], list[dict] | None],
) -> list[Sample]:
    """Stage 1, non-policy: one folded raw sample per leaf, in commit order.

    Each sample's metadata carries the leaf descriptor (``leaf``) plus the
    flat TITO bookkeeping keys. Leaves whose turns all truncate away carry no
    trainable material for any policy and are dropped here.
    """
    material: list[Sample] = []
    for leaf in state.tree.leaves():
        path = leaf.path_nodes()
        turns = compute_samples_from_openai_records(
            args,
            [node.record for node in path],
            tokenizer,
            accumulated_token_ids=leaf.token_ids,
            max_trim_tokens=max_trim_tokens,
        )
        if max_seq_len is not None:
            turns = truncate_samples_by_total_tokens(turns, max_seq_len, tokenizer)
        if not turns:
            continue
        sample = merge_samples(turns, tokenizer)
        tools = path[-1].record.request.get("tools")
        flat: dict[str, Any] = {
            "accumulated_token_ids": list(leaf.token_ids),
            _LEAF_KEY: {
                "node_id": leaf.seq,
                "parent": leaf.parent.seq if leaf.parent is not None else None,
                "path_node_ids": [n.seq for n in path],
                "response_id": leaf.response_id,
            },
        }
        mismatch = compute_mismatch(leaf.path_messages(), leaf.token_ids, tools)
        if mismatch is not None:
            flat["tito_session_mismatch"] = mismatch
        sample.metadata = {**(sample.metadata or {}), **flat}
        material.append(sample)
    return material


def default_pick(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Temporal-supersession retry trim (ruling F): a childless non-root leaf
    with a later-committed sibling was abandoned by a retry and is dropped.

    Roots are never trimmed — zero-overlap roots are independent lines
    (subagent forests are data, and a single-turn main line next to a later
    root is indistinguishable from a root retry; true root retries are rare
    because a failed first turn leaves no node). Guardrail: a superseded leaf
    longer than every later sibling's deepest leaf means the tree is not
    retry-shaped — fail loud (422) instead of guessing.
    """
    nodes = {n["id"]: n for n in session_metadata["tree"]["nodes"]}
    children: dict[int, list[int]] = {}
    for n in session_metadata["tree"]["nodes"]:
        if n["parent"] is not None:
            children.setdefault(n["parent"], []).append(n["id"])
    leaf_rows = session_metadata["tree"]["leaves"]

    kept: list[Sample] = []
    for sample in leaf_samples:
        descriptor = sample.metadata[_LEAF_KEY]
        leaf_id, parent = descriptor["node_id"], descriptor["parent"]
        later = [] if parent is None else [sibling for sibling in children[parent] if sibling > leaf_id]
        if not later:
            kept.append(sample)
            continue
        survivors_max = max(
            nodes[row["node_id"]]["num_tokens"]
            for row in leaf_rows
            if any(sibling in row["path_node_ids"] for sibling in later)
        )
        assert nodes[leaf_id]["num_tokens"] <= survivors_max, (
            f"default picker guardrail: superseded leaf (response_id={descriptor['response_id']!r}, "
            f"seq={leaf_id}) is longer than every later sibling's deepest leaf "
            f"({nodes[leaf_id]['num_tokens']} > {survivors_max} tokens) — the tree is not "
            f"retry-shaped; configure a custom sample picker to keep it"
        )
        logger.info("Default picker trimmed superseded retry leaf seq=%d", leaf_id)
    return kept


def default_merge(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Finalize the surviving set: mask each generation's completion into
    exactly one sample (earliest surviving leaf owns it), apply the semantic
    layer per sample, and assign per-branch rewards keyed by response id.

    Runs strictly after pick — ownership must be computed over the surviving
    set, or a trimmed owner would silently take its completions out of the
    training data.
    """
    nodes = {n["id"]: n for n in session_metadata["tree"]["nodes"]}
    agent = session_metadata.get("agent") or {}
    rewards = agent.get("rewards") or {}
    agent_flat = {k: v for k, v in agent.items() if k != "rewards"}

    owner: dict[int, int] = {}
    for sample in leaf_samples:
        descriptor = sample.metadata[_LEAF_KEY]
        for node_id in descriptor["path_node_ids"]:
            owner.setdefault(node_id, descriptor["node_id"])

    out: list[Sample] = []
    for sample in leaf_samples:
        descriptor = sample.metadata[_LEAF_KEY]
        response_start = len(sample.tokens) - sample.response_length
        for node_id in descriptor["path_node_ids"]:
            if owner[node_id] == descriptor["node_id"]:
                continue
            span = nodes[node_id]["completion_span"]
            lo = max(span[0] - response_start, 0)
            hi = min(span[1] - response_start, sample.response_length)
            if hi > lo:
                sample.loss_mask[lo:hi] = [0] * (hi - lo)
        # Layering: fold/base < agent semantic layer < server flat bookkeeping
        # (the agent must not be able to spoof accumulated ids or descriptors).
        server_flat = {
            k: sample.metadata[k]
            for k in ("accumulated_token_ids", "tito_session_mismatch", _LEAF_KEY)
            if k in sample.metadata
        }
        sample.metadata = {**sample.metadata, **agent_flat, **server_flat}
        if descriptor["response_id"] in rewards:
            sample.reward = rewards[descriptor["response_id"]]
        out.append(sample)
    return out
