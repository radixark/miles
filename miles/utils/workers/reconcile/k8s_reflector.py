# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.k8s_api import (
    EVENT_TYPE_ADDED,
    EVENT_TYPE_BOOKMARK,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_MODIFIED,
    KubernetesPodApi,
    PodWatchEvent,
    exception_rejects_cursor,
)
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, SourceEvent, UpsertEvent

logger = logging.getLogger(__name__)


class KubernetesReflector:
    def __init__(
        self,
        *,
        kube_client: KubernetesPodApi,
        namespace: str,
        label_selector: str,
        watch_timeout_seconds: int = 300,
        retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        assert retry_delay > 0, f"{retry_delay=} must be positive"
        assert watch_timeout_seconds > 0, f"{watch_timeout_seconds=} must be positive"

        self._kube_client = kube_client
        self._namespace = namespace
        self._label_selector = label_selector
        self._watch_timeout_seconds = watch_timeout_seconds
        self._retry_delay = retry_delay
        self._clock = clock or RealClock()

    async def watch(self) -> AsyncGenerator[SourceEvent, None]:
        cursor = _WatchCursor()
        while True:
            try:
                async with aclosing(self._watch_once(cursor)) as events:
                    async for event in events:
                        yield event
                await self._clock.sleep(self._retry_delay)
            except Exception as exception:
                if exception_rejects_cursor(exception):
                    logger.warning(f"KubernetesReflector cursor is no longer usable, relisting {cursor=}")
                    cursor.resource_version = None
                elif isinstance(exception, _UnreadableFrameError):
                    logger.error("KubernetesReflector could not read a frame, relisting", exc_info=True)
                    cursor.resource_version = None
                else:
                    logger.error("KubernetesReflector stream failed, retrying", exc_info=True)
                await self._clock.sleep(self._retry_delay)

    async def _watch_once(self, cursor: _WatchCursor) -> AsyncGenerator[SourceEvent, None]:
        if cursor.resource_version is None:
            page = await self._kube_client.list_pods(namespace=self._namespace, label_selector=self._label_selector)
            yield ReplaceEvent(objects={_pod_key(pod): pod for pod in page.pods})
            cursor.resource_version = page.resource_version

        async with aclosing(
            self._kube_client.stream_pods(
                namespace=self._namespace,
                label_selector=self._label_selector,
                resource_version=cursor.resource_version,
                timeout_seconds=self._watch_timeout_seconds,
            )
        ) as stream:
            async for raw_event in stream:
                if raw_event.type == EVENT_TYPE_ERROR:
                    if not raw_event.rejects_cursor:
                        raise RuntimeError(f"KubernetesReflector received error event {raw_event=}")
                    logger.warning(f"KubernetesReflector received a cursor error event, relisting {raw_event=}")
                    cursor.resource_version = None
                    return

                try:
                    event = _to_source_event(raw_event)
                except Exception as exception:
                    raise _UnreadableFrameError(f"a frame the cursor cannot advance past: {raw_event=}") from exception
                cursor.resource_version = raw_event.resource_version or cursor.resource_version
                if event is not None:
                    yield event


class _UnreadableFrameError(Exception):
    pass


@dataclass
class _WatchCursor:
    resource_version: str | None = None


def _to_source_event(raw_event: PodWatchEvent) -> SourceEvent | None:
    if raw_event.type in (EVENT_TYPE_ADDED, EVENT_TYPE_MODIFIED, EVENT_TYPE_DELETED):
        key = _pod_key_or_none(raw_event.obj)
        if key is None:
            return None
        if raw_event.type == EVENT_TYPE_DELETED:
            return DeleteEvent(key=key, last_obj=raw_event.obj)
        return UpsertEvent(key=key, obj=raw_event.obj)
    if raw_event.type != EVENT_TYPE_BOOKMARK:
        logger.warning(f"KubernetesReflector ignoring unknown event {raw_event.type=}")
    return None


def _pod_key_or_none(obj: Any) -> str | None:
    try:
        return _pod_key(obj)
    except Exception:
        logger.error(f"KubernetesReflector skipping a watch event whose key cannot be read {obj=}", exc_info=True)
        return None


def _pod_key(pod: Any) -> str:
    return pod.metadata.name
