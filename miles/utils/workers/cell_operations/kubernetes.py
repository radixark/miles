from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import (
    TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    BaseCellOperations,
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
)
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core.cell_view import PodIdentity
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import ContainerIdentity
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider

logger = logging.getLogger(__name__)

INJECT_FAULT_TIMEOUT_SECONDS = 60.0
TERMINATE_POLL_INTERVAL_SECONDS = 1.0


class _DeleteAttempt(enum.Enum):
    DELETED = "deleted"
    GONE = "gone"
    CONFLICT = "conflict"


class KubernetesCellOperations(BaseCellOperations):
    def __init__(self, *, provider: KubernetesWorkerProvider, namespace: str) -> None:
        self._provider = provider
        self._namespace = namespace
        self._watching: asyncio.Task[StopWatchFn] | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        await self._ensure_watching()

        wanted = set(pool_ids)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.pool_id in wanted}

    async def suspend(self, *, cell_id: str) -> None:
        await self._ensure_watching()

        pods = self._provider.pod_names_of_cell(cell_id)
        assert pods, f"cannot suspend {cell_id}, which has no pods"
        await _delete_pods(namespace=self._namespace, pod_names=pods)

    async def resume(self, *, cell_id: str) -> None:
        raise NotImplementedError(
            "a deleted cell comes back when its workload recreates it, so resume has no moment to return at"
        )

    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome:
        deadline = time.monotonic() + timeout
        try:
            return await asyncio.wait_for(
                self._terminate_observed_incarnation(
                    cell_id=cell_id, expected_workers_hash=expected_workers_hash, deadline=deadline
                ),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise CellTerminationNotConfirmedError(
                f"terminating {cell_id} ({expected_workers_hash}) did not finish within {timeout}s, "
                f"so its workers may still be running"
            ) from e

    async def _terminate_observed_incarnation(
        self, *, cell_id: str, expected_workers_hash: str, deadline: float
    ) -> CellTerminationOutcome:
        await self._ensure_watching()

        incarnation = self._provider.cell_incarnation(cell_id)
        if incarnation is None:
            logger.info("Not terminating %s: no pod of it is observed any more", cell_id)
            return CellTerminationOutcome.ALREADY_GONE
        if incarnation.workers_hash != expected_workers_hash:
            logger.warning(
                "Not terminating %s: its pods are now %s, not the %s the request was issued against",
                cell_id,
                incarnation.workers_hash,
                expected_workers_hash,
            )
            return CellTerminationOutcome.STALE

        async with _core_v1_api() as core_v1_api:
            await asyncio.gather(
                *[
                    self._delete_pod_of_incarnation(
                        core_v1_api=core_v1_api, cell_id=cell_id, pod=pod, deadline=deadline
                    )
                    for pod in incarnation.pods
                ]
            )

        await self._wait_until_pods_are_gone(cell_id=cell_id, pods=incarnation.pods, deadline=deadline)
        return CellTerminationOutcome.TERMINATED

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        await self._ensure_watching()

        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        assert (
            0 <= sub_index < len(infos)
        ), f"sub_index {sub_index} is out of range for cell {cell_id}, which has {len(infos)} workers"

        worker_name = infos[sub_index].name
        handles = self._provider.get_handles_of_worker_infos(infos)
        assert (
            worker_name in handles
        ), f"{worker_name} is not served over rpc, so no call can reach the process to crash it"

        await _inject_fault_over_rpc(handle=handles[worker_name], mode=mode, worker_name=worker_name)

    async def _delete_pod_of_incarnation(
        self, *, core_v1_api: Any, cell_id: str, pod: PodIdentity, deadline: float
    ) -> None:
        target = pod
        while True:
            if (remaining := deadline - time.monotonic()) <= 0:
                raise CellTerminationNotConfirmedError(
                    f"ran out of time asking kubernetes to delete {target.name} of {cell_id} under the "
                    f"preconditions of the incarnation this termination targeted, so it may still be running"
                )

            attempt = await asyncio.wait_for(
                _delete_pod_if_unchanged(core_v1_api=core_v1_api, namespace=self._namespace, pod=target),
                timeout=remaining,
            )
            if attempt is not _DeleteAttempt.CONFLICT:
                return

            await asyncio.sleep(TERMINATE_POLL_INTERVAL_SECONDS)
            observed = self._observed_pod(cell_id, name=target.name)
            if observed is None or _pod_incarnation_key(observed) != _pod_incarnation_key(target):
                logger.info(
                    "Pod %s of %s refused the delete and is no longer the incarnation that was targeted, so no "
                    "further delete is issued against it; whether its workers have left is decided separately",
                    target.name,
                    cell_id,
                )
                return
            target = observed

    async def _wait_until_pods_are_gone(self, *, cell_id: str, pods: list[PodIdentity], deadline: float) -> None:
        while True:
            incarnation = self._provider.cell_incarnation(cell_id)
            observed = {} if incarnation is None else {pod.name: pod for pod in incarnation.pods}
            remaining = [pod.name for pod in pods if not _pod_workers_have_left(pod, observed.get(pod.name))]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                raise CellTerminationNotConfirmedError(
                    f"pods {sorted(remaining)} of {cell_id} were asked to be deleted but are still observed, "
                    f"and nothing in what they report proves the workers of the targeted incarnation have died"
                )
            await asyncio.sleep(TERMINATE_POLL_INTERVAL_SECONDS)

    def _observed_pod(self, cell_id: str, *, name: str) -> PodIdentity | None:
        incarnation = self._provider.cell_incarnation(cell_id)
        if incarnation is None:
            return None
        return next((pod for pod in incarnation.pods if pod.name == name), None)

    async def _ensure_watching(self) -> None:
        if self._watching is None:
            self._watching = asyncio.ensure_future(self._provider.watch_cells(_ignore_cell))
        try:
            await self._watching
        except BaseException:
            self._watching = None
            raise


def _pod_incarnation_key(pod: PodIdentity) -> tuple[str, str, int]:
    return pod.name, pod.uid, pod.restart_count


def _pod_workers_have_left(target: PodIdentity, observed: PodIdentity | None) -> bool:
    if observed is None or observed.uid != target.uid:
        return True
    return _worker_container_was_replaced(target, observed)


def _worker_container_was_replaced(target: PodIdentity, observed: PodIdentity) -> bool:
    declared = _sole_declared_container_name(target)
    if declared is None or declared != _sole_declared_container_name(observed):
        return False

    before = _reported_container(target, name=declared)
    after = _reported_container(observed, name=declared)
    if before is None or after is None:
        return False
    if not before.container_id or not after.container_id:
        return False
    return after.restart_count > before.restart_count or before.container_id != after.container_id


def _sole_declared_container_name(pod: PodIdentity) -> str | None:
    if len(pod.declared_container_names) != 1:
        return None
    (name,) = pod.declared_container_names
    return name or None


def _reported_container(pod: PodIdentity, *, name: str) -> ContainerIdentity | None:
    reported = [container for container in pod.containers if container.name == name]
    if len(reported) != 1 or len(pod.containers) != 1:
        return None
    return reported[0]


async def _inject_fault_over_rpc(*, handle: BaseWorkerHandle, mode: FailureMode, worker_name: str) -> None:
    try:
        await asyncio.wait_for(
            handle.submit_without_result("inject_fault", mode=mode.value), timeout=INJECT_FAULT_TIMEOUT_SECONDS
        )
    except (WorkerUnreachableError, TimeoutError, asyncio.TimeoutError):
        logger.info("Injecting %s into %s left it unreachable, which is what was asked for", mode.value, worker_name)


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None


async def _delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    async with _core_v1_api() as core_v1_api:
        await asyncio.gather(
            *(core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names)
        )


async def _delete_pod_if_unchanged(*, core_v1_api: Any, namespace: str, pod: PodIdentity) -> _DeleteAttempt:
    from kubernetes_asyncio import client as kubernetes_client

    body = kubernetes_client.V1DeleteOptions(
        preconditions=kubernetes_client.V1Preconditions(uid=pod.uid, resource_version=pod.resource_version)
    )
    try:
        await core_v1_api.delete_namespaced_pod(name=pod.name, namespace=namespace, body=body)
    except kubernetes_client.ApiException as e:
        if e.status == 404:
            return _DeleteAttempt.GONE
        if e.status == 409:
            return _DeleteAttempt.CONFLICT
        raise
    return _DeleteAttempt.DELETED


@asynccontextmanager
async def _core_v1_api() -> AsyncIterator[Any]:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        yield kubernetes_client.CoreV1Api(api_client)
