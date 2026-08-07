from __future__ import annotations

import inspect

import pytest

from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.worker_provider.kubernetes.helm.labels import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_provider.kubernetes.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.run import KubernetesRun, PoolView
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


def _static_provider() -> SimpleWorkerProvider:
    return SimpleWorkerProvider(
        addrs={"inference-router-0-0-0": {"primary": HostAndPort(host="10.0.0.1", port=8000)}},
        cells={"inference-router-0-0": ["inference-router-0-0-0"]},
        pool_ids={"inference-router-0-0": "inference-router-0"},
    )


def _kubernetes_capability() -> KubernetesBackendCapability:
    return KubernetesBackendCapability(
        run=KubernetesRun(
            namespace="rl",
            label_selector="app.kubernetes.io/instance=r",
            pool_ids={"engine": PoolView(ports={"rpc": 8000}, worker_class=None, meta=None, ranks_per_pod=1)},
            kubernetes_client_factory=lambda: object(),
            label_keys=DEFAULT_LABEL_KEYS,
        ),
        static_provider=_static_provider(),
        cell_operations=object(),
    )


class TestKubernetesBackendCapability:
    def test_a_pool_is_answered_with_an_observer_of_that_pool(self) -> None:
        """The provider a component gets must watch its own pool_id, not whatever the namespace happens to hold."""
        capability = _kubernetes_capability()

        provider = capability.dynamic_worker_provider(pool_ids=["engine"])

        assert isinstance(provider, KubernetesWorkerProvider)
        assert provider._pools == ["engine"]

    def test_refuses_a_pool_nobody_watches(self) -> None:
        """Cells of an unwatched pool_id are never reported, so the caller would wait for them forever."""
        capability = _kubernetes_capability()

        with pytest.raises(AssertionError, match="not pool_ids of this run"):
            capability.dynamic_worker_provider(pool_ids=["engine", "trainer-actor"])

    def test_a_static_worker_is_answered_from_the_address_book(self) -> None:
        """A statically addressed worker has no cell to observe, only a predicted address."""
        capability = _kubernetes_capability()

        assert isinstance(capability.static_worker_provider(pool_id="inference-router-0"), SimpleWorkerProvider)

    def test_refuses_a_worker_the_address_book_never_heard_of(self) -> None:
        """Inventing an address would send the caller at a host that does not exist."""
        capability = _kubernetes_capability()

        with pytest.raises(AssertionError, match="not in this run's address book"):
            capability.static_worker_provider(pool_id="session-server")


class TestRayBackendCapability:
    def test_is_built_from_the_worker_manager_alone(self) -> None:
        """Under Ray there is no namespace, no label keys and no address book to read out of the environment."""
        parameters = list(inspect.signature(RayBackendCapability).parameters)

        assert parameters == ["worker_manager_handle"]

    def test_answers_both_kinds_of_request_from_that_manager(self) -> None:
        """The manager launched every worker of the run, so it knows the observed and the addressed ones alike."""
        capability = RayBackendCapability(worker_manager_handle=object())

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["engine"]), RayWorkerProvider)
        assert isinstance(capability.static_worker_provider(pool_id="inference-router-0"), RayWorkerProvider)

    def test_accepts_a_pool_no_watch_was_opened_for(self) -> None:
        """Ray resolves a name when it is asked, so nothing has to be declared up front the way a watch does."""
        capability = RayBackendCapability(worker_manager_handle=object())

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["a-pool-nobody-mentioned"]), RayWorkerProvider)


class TestDeferredBackendCapability:
    def test_builds_nothing_until_a_provider_is_asked_for(self) -> None:
        """Every served worker carries this capability, and most of them never address another worker."""
        built: list[int] = []

        DeferredBackendCapability(create=lambda: _fail_to_build(built))

        assert built == []

    def test_builds_the_backend_capability_once_and_reuses_it(self) -> None:
        """Building twice would mean two watches of the same namespace in one process."""
        built: list[int] = []
        inner = _kubernetes_capability()

        def _create() -> BackendCapability:
            built.append(1)
            return inner

        capability = DeferredBackendCapability(create=_create)

        capability.dynamic_worker_provider(pool_ids=["engine"])
        assert capability.static_worker_provider(pool_id="inference-router-0") is not None
        assert capability.cell_operations() is inner.cell_operations()
        assert built == [1]


def _fail_to_build(built: list[int]) -> BackendCapability:
    built.append(1)
    raise AssertionError("the capability was built although nobody asked for a provider")
