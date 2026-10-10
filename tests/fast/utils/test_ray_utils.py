import pytest
import ray
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils import ray_utils


def _node(index: int, *, alive: bool = True, head: bool = False) -> dict:
    resources = {"CPU": 8.0, f"node:10.0.0.{index}": 1.0}
    if head:
        resources[ray_utils._HEAD_NODE_RESOURCE] = 1.0
    return {"NodeID": fake_ray_node_id(index), "Alive": alive, "Resources": resources}


def _pinned_node_id(monkeypatch: pytest.MonkeyPatch, nodes: list[dict]) -> str:
    monkeypatch.setattr(ray, "nodes", lambda: nodes)
    strategy = ray_utils.compute_ray_pin_head_options()["scheduling_strategy"]
    assert strategy.soft is False
    return strategy.node_id


class TestComputeRayPinHeadOptions:
    def test_pins_to_the_node_carrying_the_head_resource(self, monkeypatch: pytest.MonkeyPatch):
        """The head is found through GCS, so a worker manager placed off the head still resolves it
        when the dashboard listens on 127.0.0.1 only."""
        assert _pinned_node_id(monkeypatch, [_node(1), _node(2, head=True), _node(3)]) == fake_ray_node_id(2)

    def test_a_dead_node_is_never_the_head(self, monkeypatch: pytest.MonkeyPatch):
        """ray.nodes() keeps dead entries with their resources; a hard pin to one would never schedule."""
        assert _pinned_node_id(monkeypatch, [_node(1, alive=False, head=True), _node(2, head=True)]) == (
            fake_ray_node_id(2)
        )

    def test_a_cluster_without_a_head_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """Falling back to any node would silently drop the pin the caller asked for."""
        monkeypatch.setattr(ray, "nodes", lambda: [_node(1), _node(2)])

        with pytest.raises(RuntimeError, match="head node"):
            ray_utils.compute_ray_pin_head_options()
