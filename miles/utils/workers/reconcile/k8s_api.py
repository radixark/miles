# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Protocol

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod, WatchFrame

logger = logging.getLogger(__name__)

_CURSOR_REJECTED_CODES = (410, 504)
_CURSOR_REJECTED_REASONS = ("Expired", "Gone")

EVENT_TYPE_ADDED = "ADDED"
EVENT_TYPE_MODIFIED = "MODIFIED"
EVENT_TYPE_DELETED = "DELETED"
EVENT_TYPE_BOOKMARK = "BOOKMARK"
EVENT_TYPE_ERROR = "ERROR"

POD_EVENT_TYPES = (EVENT_TYPE_ADDED, EVENT_TYPE_MODIFIED, EVENT_TYPE_DELETED)


class PodListPage(FrozenStrictBaseModel):
    pods: list[Pod]
    resource_version: str


class PodWatchEvent(FrozenStrictBaseModel):
    type: str
    pod: Pod | None
    resource_version: str | None
    rejects_cursor: bool

    @classmethod
    def from_frame(cls, *, event_type: str, obj: Any) -> PodWatchEvent:
        frame = WatchFrame.model_validate(obj)
        return cls(
            type=event_type,
            pod=Pod.model_validate(obj) if event_type in POD_EVENT_TYPES else None,
            resource_version=frame.metadata.resource_version,
            rejects_cursor=event_type == EVENT_TYPE_ERROR and _frame_rejects_cursor(frame),
        )


class KubernetesPodApi(Protocol):
    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage: ...

    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]: ...


class KubernetesAsyncioPodApi:
    def __init__(self, *, core_v1_api: Any) -> None:
        self._core_v1_api = core_v1_api

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        pod_list = await self._core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        return PodListPage(pods=list(pod_list.items), resource_version=pod_list.metadata.resource_version)

    async def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        from kubernetes_asyncio import watch as kubernetes_watch

        watcher = kubernetes_watch.Watch()
        try:
            async for event in watcher.stream(
                self._core_v1_api.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                resource_version=resource_version,
                timeout_seconds=timeout_seconds,
                allow_watch_bookmarks=True,
            ):
                yield PodWatchEvent.from_frame(event_type=event["type"], obj=event["object"])
        finally:
            await _close_quietly(watcher.close())


async def _close_quietly(closing: Any) -> None:
    try:
        await closing
    except Exception:
        logger.error("failed to close a Kubernetes watch stream", exc_info=True)


def exception_rejects_cursor(exception: BaseException) -> bool:
    return getattr(exception, "status", None) in _CURSOR_REJECTED_CODES


def _frame_rejects_cursor(frame: WatchFrame) -> bool:
    return frame.code in _CURSOR_REJECTED_CODES or frame.reason in _CURSOR_REJECTED_REASONS
