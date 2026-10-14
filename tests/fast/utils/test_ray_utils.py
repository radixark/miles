import pytest

from miles.utils import ray_utils


def _node(node_id: str, *, alive: bool = True, head: bool = False) -> dict:
    resources = {"CPU": 8.0, f"node:10.0.0.{node_id[-1]}": 1.0}
    if head:
        resources["node:__internal_head__"] = 1.0
    return {"NodeID": node_id, "Alive": alive, "Resources": resources}


class TestGetHeadNodeId:
    def test_finds_the_head_from_the_gcs_node_table(self, monkeypatch: pytest.MonkeyPatch):
        """The lookup must not need the head-only dashboard: it runs in actors placed on any node."""
        monkeypatch.setattr(ray_utils.ray, "nodes", lambda: [_node("worker1"), _node("head0", head=True)])
        assert ray_utils._get_head_node_id() == "head0"

    def test_skips_a_dead_head_entry(self, monkeypatch: pytest.MonkeyPatch):
        """A restarted head leaves its old entry behind as dead; only the live one may be pinned to."""
        nodes = [_node("old0", alive=False, head=True), _node("head1", head=True)]
        monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)
        assert ray_utils._get_head_node_id() == "head1"

    def test_raises_without_a_head(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ray_utils.ray, "nodes", lambda: [_node("worker1")])
        with pytest.raises(RuntimeError, match="head node"):
            ray_utils._get_head_node_id()
