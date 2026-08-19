from __future__ import annotations

import logging
from typing import NamedTuple

from kubernetes_asyncio import client

from miles.utils.external_utils.colocate_pairing.config import PairingConfig, PairingLayout
from miles.utils.external_utils.colocate_pairing.pods import (
    PodCoordinate,
    coordinate_of,
    gate_names,
    is_gated,
    release_patch,
)
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.loop import ReconcileLoop

logger = logging.getLogger(__name__)

_UNRELATED_KEY_PREFIX = "__unrelated__/"


class InferencePlacement(NamedTuple):
    trainer_coord: PodCoordinate
    base_gpu_id: int


class PairingController:
    _loop: ReconcileLoop

    def __init__(self, *, config: PairingConfig, core_v1: client.CoreV1Api) -> None:
        self._config = config
        self._core_v1 = core_v1
        placement_of_inference = {
            PodCoordinate(
                pool_id=pool.pool_id, cell_index=cell_index, pod_in_cell_index=pod_index
            ): _place_inference_pod(
                inference_cell_index=cell_index,
                inference_pod_index=pod_index,
                layout=pool.layout,
                trainer_pool_id=config.trainer_pool_id,
            )
            for pool in self._config.inference_pools
            for cell_index in range(pool.layout.num_inference_cells)
            for pod_index in range(pool.layout.num_pods_per_inference_cell)
        }
        self._inferences_of_trainer: dict[PodCoordinate, list[tuple[PodCoordinate, int]]] = {}
        for inference, placement in placement_of_inference.items():
            self._inferences_of_trainer.setdefault(placement.trainer_coord, []).append(
                (inference, placement.base_gpu_id)
            )

        self._trainer_key_of_coord = {
            inference: placement.trainer_coord.key for inference, placement in placement_of_inference.items()
        } | {trainer: trainer.key for trainer in self._inferences_of_trainer}

    def set_loop(self, loop: ReconcileLoop) -> None:
        self._loop = loop

    async def reconcile(self, pair_key: str) -> None:
        pods_by_coord = {
            coord: pod for pod in self._loop.get_by_parent(pair_key) if (coord := coordinate_of(pod)) is not None
        }

        trainer_coord = next((coord for coord in pods_by_coord if coord.key == pair_key), None)
        if trainer_coord is None:
            return
        gated_pods_and_base_gpu_ids = [
            (pod, base_gpu_id)
            for inference_coord, base_gpu_id in self._inferences_of_trainer.get(trainer_coord, [])
            if (pod := pods_by_coord.get(inference_coord)) is not None and is_gated(pod)
        ]
        if not gated_pods_and_base_gpu_ids:
            return

        trainer_node_name = pods_by_coord[trainer_coord].spec.node_name
        if not trainer_node_name:
            logger.info(
                "Waiting for %s to be scheduled before releasing %s",
                trainer_coord.key,
                [pod.metadata.name for pod, _ in gated_pods_and_base_gpu_ids],
            )
            return

        for inference_pod, base_gpu_id in gated_pods_and_base_gpu_ids:
            await self._release(
                inference_pod,
                node_name=trainer_node_name,
                base_gpu_id=base_gpu_id,
                trainer_key=trainer_coord.key,
            )

    async def _release(self, inference_pod: Pod, *, node_name: str, base_gpu_id: int, trainer_key: str) -> None:
        logger.info(
            "Releasing %s onto gpu %s of %s, where %s runs",
            inference_pod.metadata.name,
            base_gpu_id,
            node_name,
            trainer_key,
        )
        await self._core_v1.patch_namespaced_pod(
            name=inference_pod.metadata.name,
            namespace=self._config.namespace,
            body=release_patch(
                node_name=node_name,
                base_gpu_id=base_gpu_id,
                gates=gate_names(inference_pod),
                has_node_selector=bool(inference_pod.spec.node_selector),
                annotations=inference_pod.metadata.annotations,
            ),
        )

    def key_of(self, pod: Pod) -> str:
        if (coord := coordinate_of(pod)) is not None and (key := self._trainer_key_of_coord.get(coord)) is not None:
            return key
        return f"{_UNRELATED_KEY_PREFIX}{pod.metadata.name}"


def _place_inference_pod(
    *, inference_cell_index: int, inference_pod_index: int, layout: PairingLayout, trainer_pool_id: str
) -> InferencePlacement:
    assert 0 <= inference_cell_index < layout.num_inference_cells, f"{inference_cell_index=} outside {layout}"
    assert 0 <= inference_pod_index < layout.num_pods_per_inference_cell, f"{inference_pod_index=} outside {layout}"

    absolute_gpu = (
        layout.gpu_offset
        + (inference_cell_index * layout.num_pods_per_inference_cell + inference_pod_index)
        * layout.num_gpus_per_inference_pod
    )
    trainer_cell_index, trainer_pod_index = divmod(
        absolute_gpu // layout.num_gpus_per_node, layout.num_pods_per_trainer_cell
    )
    return InferencePlacement(
        trainer_coord=PodCoordinate(
            pool_id=trainer_pool_id, cell_index=trainer_cell_index, pod_in_cell_index=trainer_pod_index
        ),
        base_gpu_id=absolute_gpu % layout.num_gpus_per_node,
    )
