from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodListPage


def _install_fake_kubernetes_asyncio(monkeypatch: Any) -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = dict(func=None, kwargs=None, closed=0)

    class _FakeWatch:
        def stream(self, func: Any, **kwargs: Any) -> Any:
            state["func"] = func
            state["kwargs"] = kwargs
            return self

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[dict[str, Any]]:
                yield dict(type="MODIFIED", object="pod-from-the-wire")

            return _iterate()

        async def close(self) -> None:
            state["closed"] += 1

    watch_module = ModuleType("kubernetes_asyncio.watch")
    watch_module.Watch = _FakeWatch
    package = ModuleType("kubernetes_asyncio")
    package.watch = watch_module
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio.watch", watch_module)
    return watch_module, state


class TestKubernetesAsyncioPodApi:
    async def test_list_pods_delegates_to_core_v1_api(self):
        """The adapter forwards LIST to CoreV1Api.list_namespaced_pod."""
        calls: list[dict[str, Any]] = []

        class _CoreV1Api:
            async def list_namespaced_pod(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return SimpleNamespace(items=["pod-0"], metadata=SimpleNamespace(resource_version="100"))

        api = KubernetesAsyncioPodApi(core_v1_api=_CoreV1Api())
        page = await api.list_pods(namespace="ns", label_selector="a=b")

        assert page == PodListPage(pods=["pod-0"], resource_version="100")
        assert calls == [dict(namespace="ns", label_selector="a=b")]

    async def test_stream_pods_forwards_watch_options_and_closes_the_watch(self, monkeypatch):
        """The adapter drives kubernetes_asyncio's Watch and closes it, which has close() and no aclose()."""
        watch_module, state = _install_fake_kubernetes_asyncio(monkeypatch)
        _ = watch_module
        list_namespaced_pod = object()

        class _CoreV1Api:
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = list_namespaced_pod
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        events = []
        async for event in api.stream_pods(
            namespace="ns", label_selector="a=b", resource_version="42", timeout_seconds=7
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "MODIFIED"
        assert events[0].obj == "pod-from-the-wire"
        assert events[0].rejects_cursor is False
        assert state["func"] is list_namespaced_pod
        assert state["kwargs"] == dict(
            namespace="ns",
            label_selector="a=b",
            resource_version="42",
            timeout_seconds=7,
            allow_watch_bookmarks=True,
        )
        assert state["closed"] == 1

    async def test_stream_pods_closes_the_watch_when_the_consumer_stops_early(self, monkeypatch):
        """Abandoning the stream must not leak the underlying watch connection."""
        _, state = _install_fake_kubernetes_asyncio(monkeypatch)

        class _CoreV1Api:
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = object()
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        stream = api.stream_pods(namespace="ns", label_selector="a=b", resource_version="1", timeout_seconds=1)
        await stream.__anext__()
        await stream.aclose()

        assert state["closed"] == 1
