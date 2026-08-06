from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray import wiring
from miles.utils.workers.backend_capability import factory
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
        monkeypatch.setattr(factory.RayWorkerManager, "get_handle", staticmethod(lambda: object()))

        args = make_args(cluster_backend=ClusterBackend.RAY.value)

        assert isinstance(wiring.get_backend_capability(args), RayBackendCapability)

    def test_a_kubernetes_run_is_answered_by_observing_the_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under Kubernetes nothing was launched, so the capability is what observes the pods instead."""
        stub = stub_kubernetes_capability(monkeypatch)

        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)

        assert wiring.get_backend_capability(args) is stub.capability
        assert stub.specs_computed_from == [args]

    def test_the_driver_is_only_reached_when_a_ray_run_needs_its_placement_groups(self) -> None:
        """placement_group imports this module for the capability, so a top-level import back would deadlock both."""
        from miles.ray import placement_group

        assert placement_group.create_placement_groups is not None
        assert "create_placement_groups" not in vars(wiring)


@dataclass
class KubernetesCapabilityStub:
    capability: object
    specs_computed_from: list[Any] = field(default_factory=list)


def stub_kubernetes_capability(monkeypatch: pytest.MonkeyPatch) -> KubernetesCapabilityStub:
    stub = KubernetesCapabilityStub(capability=object())

    monkeypatch.setattr(wiring, "compute_specs", lambda args: stub.specs_computed_from.append(args) or [])
    monkeypatch.setattr(factory, "compute_helm_backend_capability", lambda **kwargs: stub.capability)
    monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

    return stub


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")
