from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class _ConfigException(Exception):
    pass


class _KubernetesClientProbe:
    def __init__(self) -> None:
        self.incluster_error: Exception | None = None
        self.kubeconfig_error: Exception | None = None
        self.incluster_loads = 0
        self.kubeconfig_loads = 0
        self.open_clients = 0
        self.closed_clients = 0
        self.api_client = object()
        self.core_api = SimpleNamespace(api_client=self.api_client)

    def load_incluster_config(self) -> None:
        self.incluster_loads += 1
        if self.incluster_error is not None:
            raise self.incluster_error

    async def load_kube_config(self) -> None:
        self.kubeconfig_loads += 1
        if self.kubeconfig_error is not None:
            raise self.kubeconfig_error

    async def __aenter__(self) -> object:
        self.open_clients += 1
        return self.api_client

    async def __aexit__(self, *args: Any) -> None:
        self.closed_clients += 1

    def core_v1_api(self, api_client: object) -> SimpleNamespace:
        assert api_client is self.api_client
        return self.core_api


@pytest.fixture
def kubernetes_client_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[_KubernetesClientProbe]:
    import sys

    probe = _KubernetesClientProbe()
    package = ModuleType("kubernetes_asyncio")
    package.config = SimpleNamespace(
        ConfigException=_ConfigException,
        load_incluster_config=probe.load_incluster_config,
        load_kube_config=probe.load_kube_config,
    )
    package.client = SimpleNamespace(ApiClient=lambda: probe, CoreV1Api=probe.core_v1_api)
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
    yield probe
