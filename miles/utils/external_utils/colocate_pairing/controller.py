from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from kubernetes_asyncio import client

from miles.utils.external_utils.colocate_pairing.config import PairingConfig, PairingLayout
from miles.utils.external_utils.colocate_pairing.devices import TrainerDevices, read_trainer_devices
from miles.utils.external_utils.colocate_pairing.pods import (
    PodCoordinate,
    coordinate_of,
    gate_names,
    is_gated,
    paired_trainer_uid_of,
    release_patch,
)
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.loop import ReconcileLoop

logger = logging.getLogger(__name__)

_UNRELATED_KEY_PREFIX = "__unrelated__/"


class TrainerDevicesPendingError(RuntimeError):
    """Raised while waiting for a live trainer pod to publish its assigned physical GPU UUIDs."""


class InferencePlacement(NamedTuple):
    trainer_coord: PodCoordinate
    base_gpu_id: int


class _InferenceBinding(NamedTuple):
    coord: PodCoordinate
    slot_offset: int
    num_gpus: int


class PairingController:
    _loop: ReconcileLoop

    def __init__(self, *, config: PairingConfig, core_v1: client.CoreV1Api) -> None:
        self._config = config
        self._core_v1 = core_v1
        self._expected_trainer_gpus = config.inference_pools[0].layout.gpus_per_trainer_pod
        placement_by_inference: dict[PodCoordinate, tuple[InferencePlacement, int]] = {
            PodCoordinate(pool_id=pool.pool_id, cell_index=cell_index, pod_in_cell_index=pod_index): (
                _place_inference_pod(
                    inference_cell_index=cell_index,
                    inference_pod_index=pod_index,
                    layout=pool.layout,
                    trainer_pool_id=config.trainer_pool_id,
                ),
                pool.layout.num_gpus_per_inference_pod,
            )
            for pool in self._config.inference_pools
            for cell_index in range(pool.layout.num_inference_cells)
            for pod_index in range(pool.layout.num_pods_per_inference_cell)
        }
        self._inferences_of_trainer: dict[PodCoordinate, list[_InferenceBinding]] = {}
        for inference, (placement, num_gpus) in placement_by_inference.items():
            self._inferences_of_trainer.setdefault(placement.trainer_coord, []).append(
                _InferenceBinding(coord=inference, slot_offset=placement.base_gpu_id, num_gpus=num_gpus)
            )

        self._trainer_key_of_coord = {
            inference: placement.trainer_coord.key for inference, (placement, _) in placement_by_inference.items()
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
        trainer_pod = pods_by_coord[trainer_coord]
        trainer_uid = trainer_pod.metadata.uid

        bindings = self._inferences_of_trainer.get(trainer_coord, [])
        stale_inferences = [
            pod
            for binding in bindings
            if (pod := pods_by_coord.get(binding.coord)) is not None
            and not is_gated(pod)
            and pod.metadata.deletion_timestamp is None
            and (paired_uid := paired_trainer_uid_of(pod)) is not None
            and paired_uid != trainer_uid
        ]
        if stale_inferences:
            await self._delete_stale_inferences(stale_inferences, trainer_coord=trainer_coord, trainer_uid=trainer_uid)

        if trainer_pod.metadata.deletion_timestamp is not None:
            logger.info("Waiting for terminating trainer %s to be replaced", trainer_coord.key)
            return

        gated_bindings = [
            (pod, binding)
            for binding in bindings
            if (pod := pods_by_coord.get(binding.coord)) is not None
            and pod.metadata.deletion_timestamp is None
            and is_gated(pod)
        ]
        if not gated_bindings:
            return

        trainer_node_name = trainer_pod.spec.node_name
        if not trainer_node_name:
            logger.info(
                "Waiting for %s to be scheduled before releasing %s",
                trainer_coord.key,
                [pod.metadata.name for pod, _ in gated_bindings],
            )
            return

        trainer_devices = self._require_live_trainer_devices(
            trainer_coord=trainer_coord,
            trainer_uid=trainer_uid,
            trainer_node_name=trainer_node_name,
        )

        outcomes = await asyncio.gather(
            *(
                self._release(
                    inference_pod,
                    node_name=trainer_node_name,
                    binding=binding,
                    trainer_key=trainer_coord.key,
                    trainer_uid=trainer_uid,
                    trainer_devices=trainer_devices,
                )
                for inference_pod, binding in gated_bindings
            ),
            return_exceptions=True,
        )
        for (inference_pod, _), outcome in zip(gated_bindings, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Releasing %s failed; the next reconcile of %s tries it again",
                    inference_pod.metadata.name,
                    trainer_coord.key,
                    exc_info=outcome,
                )

    def _require_live_trainer_devices(
        self,
        *,
        trainer_coord: PodCoordinate,
        trainer_uid: str,
        trainer_node_name: str,
    ) -> TrainerDevices | None:
        if self._config.pairing_dir is None:
            return None

        devices = read_trainer_devices(pairing_dir=self._config.pairing_dir, coord=trainer_coord)
        if devices is None:
            raise TrainerDevicesPendingError(
                f"Waiting for {trainer_coord.key} ({trainer_uid}) to publish physical GPU UUIDs under "
                f"{self._config.pairing_dir}"
            )
        if devices.pod_uid != trainer_uid or devices.node_name != trainer_node_name:
            raise TrainerDevicesPendingError(
                f"Pairing file for {trainer_coord.key} belongs to pod_uid={devices.pod_uid!r} on "
                f"{devices.node_name!r}, waiting for live trainer pod_uid={trainer_uid!r} on {trainer_node_name!r}"
            )
        if len(devices.gpu_uuids) != self._expected_trainer_gpus:
            raise ValueError(
                f"Trainer {trainer_coord.key} published {len(devices.gpu_uuids)} GPU UUID(s), expected "
                f"{self._expected_trainer_gpus}: {list(devices.gpu_uuids)}"
            )
        return devices

    async def _delete_stale_inferences(
        self,
        stale_pods: list[Pod],
        *,
        trainer_coord: PodCoordinate,
        trainer_uid: str,
    ) -> None:
        outcomes = await asyncio.gather(
            *(
                self._delete_stale_inference(pod, trainer_coord=trainer_coord, trainer_uid=trainer_uid)
                for pod in stale_pods
            ),
            return_exceptions=True,
        )
        for pod, outcome in zip(stale_pods, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Deleting stale inference pod %s failed; the next reconcile of %s tries it again",
                    pod.metadata.name,
                    trainer_coord.key,
                    exc_info=outcome,
                )

    async def _delete_stale_inference(
        self,
        inference_pod: Pod,
        *,
        trainer_coord: PodCoordinate,
        trainer_uid: str,
    ) -> None:
        logger.info(
            "Deleting stale inference pod %s paired with trainer uid %s; live %s has uid %s",
            inference_pod.metadata.name,
            paired_trainer_uid_of(inference_pod),
            trainer_coord.key,
            trainer_uid,
        )
        try:
            await self._core_v1.delete_namespaced_pod(
                name=inference_pod.metadata.name,
                namespace=self._config.namespace,
            )
        except client.ApiException as exc:
            if exc.status != 404:
                raise

    async def _release(
        self,
        inference_pod: Pod,
        *,
        node_name: str,
        binding: _InferenceBinding,
        trainer_key: str,
        trainer_uid: str,
        trainer_devices: TrainerDevices | None,
    ) -> None:
        if trainer_devices is not None:
            gpu_slice = trainer_devices.gpu_uuids[binding.slot_offset : binding.slot_offset + binding.num_gpus]
            nvidia_visible_devices = ",".join(gpu_slice)
            effective_base_gpu_id = 0
        else:
            nvidia_visible_devices = "all"
            effective_base_gpu_id = binding.slot_offset

        logger.info(
            "Releasing %s onto gpu %s (%s) of %s, where %s (%s) runs",
            inference_pod.metadata.name,
            effective_base_gpu_id,
            nvidia_visible_devices,
            node_name,
            trainer_key,
            trainer_uid,
        )
        await self._core_v1.patch_namespaced_pod(
            name=inference_pod.metadata.name,
            namespace=self._config.namespace,
            body=release_patch(
                node_name=node_name,
                base_gpu_id=effective_base_gpu_id,
                gates=gate_names(inference_pod),
                annotations=inference_pod.metadata.annotations,
                nvidia_visible_devices=nvidia_visible_devices,
                paired_trainer_uid=trainer_uid,
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
    trainer_pod_width = layout.gpus_per_trainer_pod
    trainer_cell_index, trainer_pod_index = divmod(absolute_gpu // trainer_pod_width, layout.num_pods_per_trainer_cell)
    return InferencePlacement(
        trainer_coord=PodCoordinate(
            pool_id=trainer_pool_id, cell_index=trainer_cell_index, pod_in_cell_index=trainer_pod_index
        ),
        base_gpu_id=absolute_gpu % trainer_pod_width,
    )
