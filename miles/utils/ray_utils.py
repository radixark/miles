import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Ray gives the head node this resource. ray.nodes() reads it from the GCS, which every node reaches; the state
# API (ray.util.state.list_nodes) asks the dashboard, which only the head serves, at 127.0.0.1:8265 by default, so
# a caller scheduled on another node (the RayWorkerManager on a multi-node cluster) cannot reach it.
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
        if node["Alive"] and _HEAD_NODE_RESOURCE in node["Resources"]:
            return node["NodeID"]
    raise RuntimeError("Could not find a head node in the Ray cluster")
