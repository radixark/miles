from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.types import ClusterBackend

NAMESPACE_ENV_VAR = "MILES_TEST_K8S_NAMESPACE"

KUBECTL_TIMEOUT_SECONDS: float = 60.0

LEADER_WORKER_SET_API_PATH: str = "/apis/leaderworkerset.x-k8s.io/v1"

REQUIRED_NAMESPACED_PERMISSIONS: tuple[tuple[str, str], ...] = (
    ("list", "pods"),
    ("delete", "pods"),
    ("list", "leaderworkersets.leaderworkerset.x-k8s.io"),
)


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str


def kubernetes_availability() -> BackendAvailability:
    return kubernetes_availability_of_namespace(
        os.environ.get(NAMESPACE_ENV_VAR, ""), namespace_source=NAMESPACE_ENV_VAR
    )


def kubernetes_availability_of_namespace(namespace: str, *, namespace_source: str) -> BackendAvailability:
    for tool, why in (("kubectl", "so no cluster can be reached"), ("helm", "and a run is installed as a release")):
        if shutil.which(tool) is None:
            return BackendAvailability(False, f"{tool} is not installed, {why}")

    if not namespace:
        return BackendAvailability(
            False,
            f"set {namespace_source} to a namespace of your own; see docs/advanced/cluster-backend.md",
        )

    reachable = _run_kubectl(["get", "--raw", "/version"])
    if reachable.returncode != 0:
        return BackendAvailability(False, f"the cluster refused the credentials: {reachable.stderr.strip()[:200]}")

    served = _run_kubectl(["get", "--raw", LEADER_WORKER_SET_API_PATH])
    if served.returncode != 0:
        return BackendAvailability(False, "LeaderWorkerSet is not served by this cluster; an admin has to add it")

    for verb, resource in REQUIRED_NAMESPACED_PERMISSIONS:
        allowed = _run_kubectl(["auth", "can-i", verb, resource, "--namespace", namespace])
        if allowed.stdout.strip() != "yes":
            return BackendAvailability(False, f"this account may not {verb} {resource} in namespace {namespace}")

    return BackendAvailability(True, f"using namespace {namespace}")


def ray_availability() -> BackendAvailability:
    try:
        import ray  # noqa: F401
    except ImportError:
        return BackendAvailability(False, "ray is not installed")
    return BackendAvailability(True, "ray is importable")


def _run_kubectl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run_process(["kubectl", *args], capture_output=True, check=False, timeout=KUBECTL_TIMEOUT_SECONDS)


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
