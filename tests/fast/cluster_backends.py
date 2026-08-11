from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest

from miles.utils.workers.types import ClusterBackend

NAMESPACE_ENV_VAR = "MILES_TEST_K8S_NAMESPACE"


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str


def kubernetes_availability() -> BackendAvailability:
    if shutil.which("kubectl") is None:
        return BackendAvailability(False, "kubectl is not installed, so no cluster can be reached")
    if shutil.which("helm") is None:
        return BackendAvailability(False, "helm is not installed, and a run is installed as a release")
    if not (namespace := os.environ.get(NAMESPACE_ENV_VAR)):
        return BackendAvailability(
            False,
            f"set {NAMESPACE_ENV_VAR} to a namespace of your own; see docs/advanced/cluster-backend.md",
        )

    reachable = subprocess.run(["kubectl", "get", "--raw", "/version"], capture_output=True, text=True)
    if reachable.returncode != 0:
        return BackendAvailability(False, f"the cluster refused the credentials: {reachable.stderr.strip()[:200]}")

    crd = subprocess.run(
        ["kubectl", "get", "crd", "leaderworkersets.leaderworkerset.x-k8s.io"], capture_output=True, text=True
    )
    if crd.returncode != 0:
        return BackendAvailability(False, "LeaderWorkerSet is not installed; an admin has to add it")

    return BackendAvailability(True, f"using namespace {namespace}")


def ray_availability() -> BackendAvailability:
    try:
        import ray  # noqa: F401
    except ImportError:
        return BackendAvailability(False, "ray is not installed")
    return BackendAvailability(True, "ray is importable")


_AVAILABILITY = {
    ClusterBackend.RAY.value: ray_availability,
    ClusterBackend.KUBERNETES.value: kubernetes_availability,
}


def require_backend(backend: str) -> str:
    availability = _AVAILABILITY[backend]()
    if not availability.available:
        pytest.skip(f"{backend} backend unavailable: {availability.reason}")
    return os.environ.get(NAMESPACE_ENV_VAR, "")


def both_backends(test):
    return pytest.mark.parametrize("cluster_backend", sorted(_AVAILABILITY), ids=sorted(_AVAILABILITY))(test)
