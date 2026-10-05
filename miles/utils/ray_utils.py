import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Ray marks the head node with this resource. Reading it through ray.nodes() goes to GCS, so the
# lookup works from any node; ray.util.state.list_nodes() asks the dashboard, which a head started
# without --dashboard-host serves on 127.0.0.1 only.
_HEAD_NODE_RESOURCE = "node:__internal_head__"


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    for node in ray.nodes():
        if node.get("Alive") and _HEAD_NODE_RESOURCE in node.get("Resources", {}):
            return node["NodeID"]
    raise RuntimeError("Could not find a head node in the Ray cluster")
