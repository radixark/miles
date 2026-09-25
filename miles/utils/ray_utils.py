import ray
from ray._common.constants import HEAD_NODE_RESOURCE_NAME
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options(head_node_id: str | None = None):
    if head_node_id is None:
        head_node_id = get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def get_head_node_id() -> str:
    for node in ray.nodes():
        if node.get("Alive") and HEAD_NODE_RESOURCE_NAME in node.get("Resources", {}):
            return node["NodeID"]
    raise RuntimeError("Could not find a head node in the Ray cluster")
