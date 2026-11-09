from __future__ import annotations

from typing import TYPE_CHECKING

from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.session.v2.tree_trajectory import SessionTree

SESSION_ROLLOUT_METRICS_KEY = "session_rollout_metrics"


def build_session_rollout_metrics(session_id: str, tree: SessionTree, *, with_spec_info: bool) -> dict:
    metrics: dict = {"tree": {"roots": len(tree.roots), "leaves": len(tree.leaves()), "turns": len(tree.nodes)}}
    if with_spec_info:
        spec_info = Sample.SpecInfo()
        for node in tree.nodes:
            spec_info.add(node.record.response["choices"][0]["meta_info"])
        metrics["spec_info"] = spec_info.to_dict()
    return {"session_id": session_id, "metrics": metrics}
