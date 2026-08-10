from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_provider import FakePodApi
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import RELEASE, make_engine_spec, make_router_spec

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_provider.static import StaticWorkerProvider

NAMESPACE = "team-a"
ROUTER_HOST = naming.static_worker_host(RELEASE, "inference-router-0", 0)
RAY_WORKER_MANAGER_MODULE = "miles.utils.workers.ray_worker_manager"


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    recorded: list[tuple[str, list[str]]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        recorded.append((namespace, list(pod_names)))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_create_kubernetes_client", fake_pod_api.installed)


@pytest.fixture(autouse=True)
def _release_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(RELEASE_ENV_VAR, RELEASE)


def install_workers(*, pods: list[Any] | None = None) -> KubernetesBackendCapability:
    fake_pod_api.install(FakePodApi(pods=list(pods or [])))

    return compute_helm_backend_capability(specs=[make_router_spec(), make_engine_spec()])


class TestKubernetesAssembly:
    def test_components_of_the_process_then_see_a_kubernetes_provider(self) -> None:
        """The whole point of the assembly: the capability must stop answering with Ray."""
        capability = install_workers()

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["engine"]), KubernetesWorkerProvider)

    def test_a_static_worker_resolves_to_the_address_the_launcher_decided(self) -> None:
        """A router has no cell to observe, so its address is recomputed from the release that deployed it."""
        capability = install_workers()

        provider = capability.static_worker_provider(pool_id="inference-router-0")
        addr = asyncio.run(provider.get_addrs("inference-router-0-0-0"))["primary"]

        assert addr.host == ROUTER_HOST
        assert addr.port == 8000

    def test_refuses_a_static_worker_the_run_never_deployed(self) -> None:
        """Answering with an invented address would send the caller at nothing at all."""
        capability = install_workers()

        with pytest.raises(AssertionError, match="not a static pool"):
            capability.static_worker_provider(pool_id="inference-router-9")

    def test_the_static_scope_answers_with_the_address_book_rather_than_with_the_watcher(self) -> None:
        """Statically addressed components need no watch, and a watch would never report them anyway."""
        capability = install_workers()

        static = capability.static_worker_provider(pool_id="inference-router-0")

        assert isinstance(static, StaticWorkerProvider)
        with pytest.raises(NotImplementedError, match="does not observe cells"):
            asyncio.run(static.watch_cells(lambda cell_id, info: None))

    def test_every_component_observes_the_same_run(self) -> None:
        """Each component watches for itself, but they must all be told about the same pods."""
        capability = install_workers()

        first = capability.dynamic_worker_provider(pool_ids=["engine"])
        second = capability.dynamic_worker_provider(pool_ids=["engine"])

        assert first._run is second._run

    def test_suspending_a_cell_deletes_its_pods(self, deleted) -> None:
        """Kubernetes has no suspend: a cell heals because its workload recreates deleted pods."""
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])
        operations = capability.cell_operations()

        asyncio.run(operations.suspend(cell_id="engine-0"))

        assert deleted == [(NAMESPACE, ["engine-0-0"])]

    def test_listing_cells_starts_the_watch_it_needs(self) -> None:
        """The api server asks for cells without knowing that observation has to be started first."""
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])
        operations = capability.cell_operations()

        infos = asyncio.run(operations.cell_infos(pool_ids=["engine"]))

        assert list(infos) == ["engine-0"]

    def test_never_reaches_for_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch, deleted) -> None:
        """A namespace has no Ray cluster, so touching the manager would fail the run there."""
        monkeypatch.setitem(sys.modules, RAY_WORKER_MANAGER_MODULE, None)
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])

        operations = capability.cell_operations()
        asyncio.run(operations.suspend(cell_id="engine-0"))

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["engine"]), KubernetesWorkerProvider)
        assert deleted == [(NAMESPACE, ["engine-0-0"])]
