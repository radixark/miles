from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from kubernetes_asyncio import client

from tests.utils.soak.k8s_utils import pod_manipulation
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget


# ============================== fake CoreV1 ==============================


def _pod_target(**overrides: object) -> SoakPodTarget:
    return SoakPodTarget(**{"namespace": "ns", "release": "rel", "name": "pod-a", "uid": "uid-a", **overrides})


def _live_pod(uid: str, *, resource_version: str | None = "rv-7", deleting: bool = False) -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(
            uid=uid, resource_version=resource_version, deletion_timestamp="2026-09-26T00:00:00Z" if deleting else None
        )
    )


def _api_error(status: int) -> client.ApiException:
    return client.ApiException(status=status, reason="scripted")


class _FakePodApi:
    def __init__(
        self, *, reads: list[client.V1Pod | BaseException], delete_error: BaseException | None = None
    ) -> None:
        self._reads = reads
        self._delete_error = delete_error
        self.calls: list[tuple[str, dict]] = []

    async def read_namespaced_pod(self, *, name: str, namespace: str) -> client.V1Pod:
        self.calls.append(("read", {"name": name, "namespace": namespace}))
        reply = self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def delete_namespaced_pod(self, *, name: str, namespace: str, body: client.V1DeleteOptions) -> None:
        self.calls.append(("delete", {"name": name, "namespace": namespace, "body": body}))
        if self._delete_error is not None:
            raise self._delete_error


def _patch_pod_api(monkeypatch: pytest.MonkeyPatch, api: _FakePodApi) -> None:
    @asynccontextmanager
    async def _core_v1_api() -> AsyncIterator[_FakePodApi]:
        yield api

    monkeypatch.setattr(pod_manipulation, "core_v1_api", _core_v1_api)
