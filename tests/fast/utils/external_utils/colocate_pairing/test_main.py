from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes_asyncio import watch as kubernetes_watch
from pydantic import ValidationError

from miles.utils.external_utils.colocate_pairing import __main__ as main_module
from miles.utils.external_utils.colocate_pairing.config import PairingConfig


def _config_json() -> str:
    return """{
        "namespace": "training",
        "release": "example",
        "trainer_pool_id": "trainer",
        "inference_pools": [{
            "pool_id": "inference",
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "num_gpus_per_inference_pod": 8,
                "gpu_offset": 0
            }
        }]
    }"""


def _config() -> PairingConfig:
    return PairingConfig.model_validate_json(_config_json())


class TestMain:
    def test_main_rejects_a_rendered_config_missing_required_fields(self) -> None:
        """A rendered config missing required fields is rejected before startup."""
        with pytest.raises(ValidationError, match="release"):
            main_module.main(["--config", '{"namespace": "training"}'])


class TestRunForever:
    async def test_run_forever_falls_back_to_the_kubeconfig_when_not_in_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unavailable in-cluster config falls back before startup continues."""
        calls: list[str] = []

        def load_incluster_config() -> None:
            calls.append("incluster")
            raise main_module.kube_config.ConfigException()

        async def load_kube_config() -> None:
            calls.append("kubeconfig")

        class StartupReached(Exception):
            pass

        class FakeApiClient:
            async def __aenter__(self) -> Any:
                raise StartupReached

            async def __aexit__(self, *args: Any) -> None:
                return None

        monkeypatch.setattr(main_module.kube_config, "load_incluster_config", load_incluster_config)
        monkeypatch.setattr(main_module.kube_config, "load_kube_config", load_kube_config)
        monkeypatch.setattr(main_module.client, "ApiClient", FakeApiClient)

        with pytest.raises(StartupReached):
            await main_module._run_forever(_config())

        assert calls == ["incluster", "kubeconfig"]

    async def test_run_forever_starts_kubernetes_discovery_and_waits_until_terminated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup lists the configured pods before entering its terminal wait."""
        boundary = _install_kubernetes_boundary(monkeypatch)

        task = asyncio.create_task(main_module._run_forever(_config()))
        await asyncio.wait_for(boundary.discovery_started.wait(), timeout=1.0)

        assert boundary.list_calls == [
            {"namespace": "training", "label_selector": "app.kubernetes.io/instance=example"}
        ]
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _install_kubernetes_boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    boundary = SimpleNamespace(list_calls=[], discovery_started=asyncio.Event())

    class FakeApiClient:
        async def __aenter__(self) -> FakeApiClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    class FakeCoreV1Api:
        def __init__(self, api_client: FakeApiClient) -> None:
            self.api_client = api_client

        async def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> SimpleNamespace:
            boundary.list_calls.append({"namespace": namespace, "label_selector": label_selector})
            boundary.discovery_started.set()
            return SimpleNamespace(items=[], metadata=SimpleNamespace(resource_version="1"))

    class FakeWatch:
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            await asyncio.Future()
            yield

        async def close(self) -> None:
            return None

    monkeypatch.setattr(main_module.kube_config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(main_module.client, "ApiClient", FakeApiClient)
    monkeypatch.setattr(main_module.client, "CoreV1Api", FakeCoreV1Api)
    monkeypatch.setattr(kubernetes_watch, "Watch", FakeWatch)
    return boundary
