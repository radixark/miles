from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.source_scan import REPO_ROOT

from miles.ray import wiring
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


async def _resolved(events: list[object], event: object) -> None:
    events.append(event)


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


class TestShutdownWorkerManager:
    async def test_a_ray_run_shuts_down_and_kills_its_own_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver must release the exact manager it launched instead of sweeping processes by name."""
        events: list[object] = []
        manager = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: _resolved(events, "shutdown")))
        monkeypatch.setattr(wiring.ray, "kill", lambda handle: events.append(("kill", handle)))

        await wiring.shutdown_worker_manager(manager)

        assert events == ["shutdown", ("kill", manager)]

    async def test_a_kubernetes_run_has_no_ray_manager_to_shut_down(self) -> None:
        """The shared driver cleanup remains a no-op when Kubernetes owns worker lifecycles."""
        await wiring.shutdown_worker_manager(None)

    @pytest.mark.parametrize(
        "script",
        sorted(path for path in REPO_ROOT.glob("train*.py") if "launch_worker_manager" in path.read_text()),
        ids=lambda path: path.name,
    )
    def test_every_finite_driver_shuts_down_the_manager_it_launched(self, script: Path) -> None:
        """Every driver that owns a worker manager must release that same manager on its normal return path."""
        calls = [
            node
            for node in ast.walk(ast.parse(script.read_text(), filename=str(script)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "shutdown_worker_manager"
        ]

        assert len(calls) == 1
        assert ast.unparse(calls[0]) == "shutdown_worker_manager(_worker_manager)"


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


class TestTheWireTheWorkerManagerIsLaunchedWith:
    @pytest.mark.parametrize("resolved", ["ray", "rpc"])
    def test_the_resolved_wire_reaches_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch, resolved: str
    ) -> None:
        """The flag only means anything if the wire validation resolved reaches the manager it starts."""
        from miles.ray import placement_group

        launched: list[WorkerCommBackend] = []
        monkeypatch.setattr(wiring, "compute_specs", lambda args: [])
        monkeypatch.setattr(placement_group, "create_placement_groups", lambda args: {})
        monkeypatch.setattr(
            wiring.RayWorkerManager,
            "launch",
            staticmethod(lambda args, specs, pgs, *, comm_backend: launched.append(comm_backend)),
        )

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value, worker_comm_backend=resolved)
        wiring._launch_ray_worker_manager(args)

        assert launched == [WorkerCommBackend(resolved)]


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
