from __future__ import annotations

from collections.abc import Mapping

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

_GATE_NAME = "miles.radixark.io/colocate-pairing"

_NODE_NAME_FIELD = "metadata.name"

NVIDIA_VISIBLE_DEVICES_ANNOTATION = "miles.radixark.io/nvidia-visible-devices"
PAIRED_TRAINER_UID_ANNOTATION = "miles.radixark.io/paired-trainer-uid"


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


def paired_trainer_uid_of(pod: Pod) -> str | None:
    return pod.metadata.annotations.get(PAIRED_TRAINER_UID_ANNOTATION)


def release_patch(
    *,
    node_name: str,
    base_gpu_id: int,
    gates: list[str],
    annotations: Mapping[str, str],
    nvidia_visible_devices: str = "all",
    paired_trainer_uid: str | None = None,
) -> list[dict[str, object]]:
    key = DEFAULT_LABEL_KEYS.base_gpu_id_annotation
    assert annotations, (
        f"the pod carries no annotations, so adding {key} under a map that does not exist is a patch "
        f"the apiserver refuses"
    )
    assert nvidia_visible_devices, "nvidia_visible_devices must be non-empty when releasing a colocated inference pod"

    index = gates.index(_GATE_NAME)
    pin = {
        "op": "add",
        "path": "/spec/affinity",
        "value": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchFields": [{"key": _NODE_NAME_FIELD, "operator": "In", "values": [node_name]}]}
                    ]
                }
            }
        },
    }
    patch: list[dict[str, object]] = [
        pin,
        {"op": "add", "path": f"/metadata/annotations/{_escape_pointer(key)}", "value": str(base_gpu_id)},
        {
            "op": "add",
            "path": f"/metadata/annotations/{_escape_pointer(NVIDIA_VISIBLE_DEVICES_ANNOTATION)}",
            "value": nvidia_visible_devices,
        },
    ]
    if paired_trainer_uid is not None:
        patch.append(
            {
                "op": "add",
                "path": f"/metadata/annotations/{_escape_pointer(PAIRED_TRAINER_UID_ANNOTATION)}",
                "value": paired_trainer_uid,
            }
        )
    patch.extend(
        [
            {"op": "test", "path": f"/spec/schedulingGates/{index}/name", "value": _GATE_NAME},
            {"op": "remove", "path": f"/spec/schedulingGates/{index}"},
        ]
    )
    return patch


def gate_names(pod: Pod) -> list[str]:
    return [gate.name for gate in pod.spec.scheduling_gates]


def is_gated(pod: Pod) -> bool:
    return _GATE_NAME in gate_names(pod)


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")
