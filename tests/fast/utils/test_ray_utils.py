import pytest
from ray._common.constants import HEAD_NODE_RESOURCE_NAME

import miles.utils.ray_utils as ray_utils


def test_get_head_node_id_reads_the_live_head_from_ray_core(monkeypatch):
    nodes = [
        {"NodeID": "dead-head", "Alive": False, "Resources": {HEAD_NODE_RESOURCE_NAME: 1.0}},
        {"NodeID": "worker", "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "head", "Alive": True, "Resources": {HEAD_NODE_RESOURCE_NAME: 1.0}},
    ]
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)

    assert ray_utils.get_head_node_id() == "head"


def test_get_head_node_id_rejects_a_cluster_without_a_live_head(monkeypatch):
    monkeypatch.setattr(
        ray_utils.ray,
        "nodes",
        lambda: [{"NodeID": "worker", "Alive": True, "Resources": {"CPU": 8.0}}],
    )

    with pytest.raises(RuntimeError, match="Could not find a head node"):
        ray_utils.get_head_node_id()
