from __future__ import annotations

from typing import TYPE_CHECKING

from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.session.v2.tree_trajectory import TrajectoryNode

SESSION_ROLLOUT_METRICS_KEY = "session_rollout_metrics"


def build_session_rollout_metrics(session_id: str, nodes: list[TrajectoryNode]) -> dict:
    spec_info = Sample.SpecInfo()
    for node in nodes:
        spec_info.add(node.record.response["choices"][0]["meta_info"])
    return {
        "session_id": session_id,
        "metrics": {"spec_info": spec_info.to_dict()},
    }
