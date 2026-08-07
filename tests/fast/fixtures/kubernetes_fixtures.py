from __future__ import annotations

from typing import Any

from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_provider import FakePodApi
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import RELEASE, make_engine_spec, make_router_spec

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_capability
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_worker_host

NAMESPACE = "team-a"
ROUTER_HOST = static_worker_host(RELEASE, "inference-router-0", 0)


def install_workers(*, pods: list[Any] | None = None) -> KubernetesBackendCapability:
    fake_pod_api.install(FakePodApi(pods=list(pods or [])))

    return compute_capability(
        specs=[make_router_spec(), make_engine_spec()],
        namespace=NAMESPACE,
        release=RELEASE,
    )
