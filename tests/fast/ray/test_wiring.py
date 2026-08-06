from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes.test_assembly import install_workers

from miles.ray import wiring
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend


class TestLaunchWorkerManager:
    def test_a_ray_run_launches_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Ray path is the one every existing run takes, and it must be untouched."""
        launched: list[Any] = []
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", lambda args: launched.append(args))

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)
        wiring.launch_worker_manager(args)

        assert launched == [args]

    def test_a_kubernetes_run_launches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under Kubernetes the pods already exist, so launching actors would double the run."""
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        assert wiring.launch_worker_manager(SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)) is None


class TestGetBackendCapability:
    def test_a_ray_run_is_answered_from_the_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The manager was launched by the driver's own first line; the capability only looks it up."""
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)
        monkeypatch.setattr(wiring.RayWorkerManager, "get_handle", staticmethod(lambda: object()))

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)

        assert isinstance(wiring.get_backend_capability(args), RayBackendCapability)

    def test_a_kubernetes_run_is_answered_by_observing_the_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under Kubernetes nothing was launched, so the capability is what observes the pods instead."""
        installed: list[Any] = []
        sentinel = object()
        monkeypatch.setattr(
            wiring, "_kubernetes_backend_capability_from_args", lambda args: installed.append(args) or sentinel
        )
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)

        assert wiring.get_backend_capability(args) is sentinel
        assert installed == [args]


class TestCreateWorkerBackendCapability:
    def test_a_worker_process_builds_its_capability_only_when_something_asks_for_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every served worker builds this context, and most specs never look at it."""
        attached: list[int] = []

        def _install(args):
            attached.append(args.rollout_num_gpus)
            return install_workers(deleted=[])

        monkeypatch.setattr(wiring, "_create_backend_capability", _install)

        capability = wiring.create_worker_backend_capability(worker_argv=["--rollout-num-gpus", "8"])
        assert attached == []

        capability.dynamic_worker_provider(pool_ids=["engine"])
        capability.dynamic_worker_provider(pool_ids=["engine"])

        assert attached == [8]


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")
