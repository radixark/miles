from __future__ import annotations

import logging

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


class PairingController:
    _loop: ReconcileLoop

    def __init__(self, *, config: PairingConfig, core_v1: client.CoreV1Api) -> None:
        self._config = config
        self._core_v1 = core_v1
        self._trainer_of_inference = {
            PodCoordinate(
                pool_id=pool.pool_id, cell_index=cell_index, pod_in_cell_index=pod_index
            ): _target_trainer_pod(
                inference_cell_index=cell_index,
                inference_pod_index=pod_index,
                layout=pool.layout,
                trainer_pool_id=config.trainer_pool_id,
            )
            for pool in self._config.inference_pools
            for cell_index in range(pool.layout.num_inference_cells)
            for pod_index in range(pool.layout.num_pods_per_inference_cell)
        }
        self._inference_of_trainer = {trainer: inference for inference, trainer in self._trainer_of_inference.items()}
        assert len(self._inference_of_trainer) == len(self._trainer_of_inference), (
            f"Two inference pods target the same trainer pod under {[pool.layout for pool in self._config.inference_pools]}, "
            f"so one of them would never be woken by the trainer it waits on"
        )

    def set_loop(self, loop: ReconcileLoop) -> None:
        self._loop = loop

    async def reconcile(self, pair_key: str) -> None:
        pods_by_coord = {
            coord: pod for pod in self._loop.get_by_parent(pair_key) if (coord := coordinate_of(pod)) is not None
        }

        inference_coord = next((coord for coord in pods_by_coord if coord.key == pair_key), None)
        if inference_coord is None or not is_gated(inference_pod := pods_by_coord[inference_coord]):
            return

        trainer_coord = self._trainer_of_inference[inference_coord]
        trainer_pod = pods_by_coord.get(trainer_coord)
        trainer_node_name = trainer_pod.spec.node_name if trainer_pod is not None else None
        if not trainer_node_name:
            logger.info(
                "Waiting for %s to be scheduled before releasing %s",
                trainer_coord.key,
                inference_pod.metadata.name,
            )
            return

        logger.info(
            "Releasing %s onto %s, where %s runs",
            inference_pod.metadata.name,
            trainer_node_name,
            trainer_coord.key,
        )
        await self._core_v1.patch_namespaced_pod(
            name=inference_pod.metadata.name,
            namespace=self._config.namespace,
            body=release_patch(
                node_name=trainer_node_name,
                gates=gate_names(inference_pod),
                has_node_selector=bool(inference_pod.spec.node_selector),
            ),
        )

    def key_of(self, pod: Pod) -> str:
        if (coord := coordinate_of(pod)) is not None:
            if coord in self._trainer_of_inference:
                return coord.key
            if (inference_coord := self._inference_of_trainer.get(coord)) is not None:
                return inference_coord.key
        return f"{_UNRELATED_KEY_PREFIX}{pod.metadata.name}"


def _target_trainer_pod(
    *, inference_cell_index: int, inference_pod_index: int, layout: PairingLayout, trainer_pool_id: str
) -> PodCoordinate:
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
    return PodCoordinate(pool_id=trainer_pool_id, cell_index=trainer_cell_index, pod_in_cell_index=trainer_pod_index)
