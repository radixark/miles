import pytest
from ray._common.constants import HEAD_NODE_RESOURCE_NAME
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

import miles.utils.ray_utils as ray_utils


def test_compute_ray_pin_head_options_uses_the_live_head(monkeypatch):
    dead_head_id = fake_ray_node_id(1)
    worker_id = fake_ray_node_id(2)
    head_id = fake_ray_node_id(3)
    nodes = [
        {"NodeID": dead_head_id, "Alive": False, "Resources": {HEAD_NODE_RESOURCE_NAME: 1.0}},
        {"NodeID": worker_id, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": head_id, "Alive": True, "Resources": {HEAD_NODE_RESOURCE_NAME: 1.0}},
    ]
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)

    strategy = ray_utils.compute_ray_pin_head_options()["scheduling_strategy"]

    assert strategy.node_id == head_id
    assert strategy.soft is False


def test_compute_ray_pin_head_options_rejects_a_cluster_without_a_live_head(monkeypatch):
    monkeypatch.setattr(
        ray_utils.ray,
        "nodes",
        lambda: [{"NodeID": fake_ray_node_id(1), "Alive": True, "Resources": {"CPU": 8.0}}],
    )

    with pytest.raises(RuntimeError, match="Could not find a head node"):
        ray_utils.compute_ray_pin_head_options()
