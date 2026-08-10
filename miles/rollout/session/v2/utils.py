import logging
from argparse import Namespace
from typing import Any

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.errors import TokenizationError
from miles.rollout.session.samples.merge import (
    compute_samples_from_openai_records,
    merge_samples_with_addition_r3,
    truncate_samples_by_total_tokens,
)
from miles.rollout.session.v2.session_state import SessionRegistryV2, SessionStateV2
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def tree_metadata(state: SessionStateV2) -> dict:
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
    state: SessionStateV2,
    registry: SessionRegistryV2,
    *,
    session_id: str,
    max_seq_len: int | None,
    use_addition_r3: bool = False,
) -> list[Sample]:
    """Merge each leaf's root-to-leaf node records into one raw sample, in commit order.

    Leaves whose turns all truncate away are dropped. Each sample's metadata
    carries the ``leaf`` descriptor plus the flat TITO bookkeeping keys for
    the downstream pick/post-process hooks.

    With ``use_addition_r3``, each record along a path carries only its
    additional R3 rows; the per-leaf assembler materializes the required prefix
    because a path is the linear record chain its offsets were computed on.
    """
    material: list[Sample] = []
    for leaf in state.tree.leaves():
        path = leaf.path_nodes()
        records = [node.record for node in path]
        turns = compute_samples_from_openai_records(
            args,
            records,
            registry.tokenizer,
            accumulated_token_ids=leaf.token_ids,
            max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            use_addition_r3=use_addition_r3,
        )
        if max_seq_len is not None:
            turns = truncate_samples_by_total_tokens(turns, max_seq_len, registry.tokenizer)
        if not turns:
            continue
        if use_addition_r3:
            sample = merge_samples_with_addition_r3(args, turns, records, registry.tokenizer)
        else:
            sample = merge_samples(turns, registry.tokenizer)
        tools = path[-1].record.request.get("tools")
        flat: dict[str, Any] = {
            "accumulated_token_ids": list(leaf.token_ids),
            "leaf": {
                "node_id": leaf.seq,
                "parent": leaf.parent.seq if leaf.parent is not None else None,
                "path_node_ids": [n.seq for n in path],
                "response_id": leaf.response_id,
            },
        }
        try:
            mismatch = registry.compute_mismatch(leaf.path_messages(), leaf.token_ids, tools)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            flat["tito_session_mismatch"] = mismatch
        sample.metadata = {**(sample.metadata or {}), **flat}
        material.append(sample)
    return material
