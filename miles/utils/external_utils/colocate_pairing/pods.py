from __future__ import annotations

from collections.abc import Mapping

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

_GATE_NAME = "miles.radixark.io/colocate-pairing"

_HOSTNAME_LABEL = "kubernetes.io/hostname"


class PodCoordinate(FrozenStrictBaseModel):
    pool_id: str
    cell_index: int
    pod_in_cell_index: int

    @property
    def key(self) -> str:
        return f"{self.pool_id}/{self.cell_index}/{self.pod_in_cell_index}"


def coordinate_of(pod: Pod) -> PodCoordinate | None:
    parsed = pod_view.parse_pod(pod, DEFAULT_LABEL_KEYS)
    if parsed is None:
        return None
    return PodCoordinate(
        pool_id=parsed.pool_id, cell_index=parsed.cell_index, pod_in_cell_index=parsed.pod_in_cell_index
    )


def release_patch(
    *,
    node_name: str,
    base_gpu_id: int,
    gates: list[str],
    has_node_selector: bool,
    annotations: Mapping[str, str],
) -> list[dict[str, object]]:
    key = DEFAULT_LABEL_KEYS.base_gpu_id_annotation
    assert annotations, (
        f"the pod carries no annotations, so adding {key} under a map that does not exist is a patch "
        f"the apiserver refuses"
    )

    index = gates.index(_GATE_NAME)
    pin = (
        {"op": "add", "path": f"/spec/nodeSelector/{_escape_pointer(_HOSTNAME_LABEL)}", "value": node_name}
        if has_node_selector
        else {"op": "add", "path": "/spec/nodeSelector", "value": {_HOSTNAME_LABEL: node_name}}
    )
    return [
        pin,
        {"op": "add", "path": f"/metadata/annotations/{_escape_pointer(key)}", "value": str(base_gpu_id)},
        {"op": "test", "path": f"/spec/schedulingGates/{index}/name", "value": _GATE_NAME},
        {"op": "remove", "path": f"/spec/schedulingGates/{index}"},
    ]


def gate_names(pod: Pod) -> list[str]:
    return [gate.name for gate in pod.spec.scheduling_gates]


def is_gated(pod: Pod) -> bool:
    return _GATE_NAME in gate_names(pod)


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")
