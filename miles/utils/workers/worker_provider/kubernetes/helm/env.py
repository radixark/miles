from __future__ import annotations

import os
from pathlib import Path

from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys

INSTANCE_LABEL = "app.kubernetes.io/instance"

DEFAULT_LABEL_KEYS = CellLabelKeys(
    pool_id="miles.radixark.io/pool",
    cell_index="leaderworkerset.sigs.k8s.io/group-index",
    pod_in_cell_index="leaderworkerset.sigs.k8s.io/worker-index",
    cell_size_annotation="leaderworkerset.sigs.k8s.io/size",
    meta_annotation_prefix="miles.radixark.io/meta-",
    gpu_ids_meta="gpu_ids",
)

NAMESPACE_ENV_VAR = "MILES_K8S_NAMESPACE"
RELEASE_ENV_VAR = "MILES_K8S_RELEASE"
NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def current_namespace() -> str:
    if namespace := os.environ.get(NAMESPACE_ENV_VAR, ""):
        return namespace
    assert NAMESPACE_FILE.exists(), (
        f"the driver runs outside a pod, so it cannot tell which namespace holds its workers: "
        f"set {NAMESPACE_ENV_VAR}"
    )
    namespace = NAMESPACE_FILE.read_text().strip()
    assert namespace, f"{NAMESPACE_FILE} is empty, so no namespace can be observed"
    return namespace


def current_release() -> str:
    release = os.environ.get(RELEASE_ENV_VAR, "")
    assert release, (
        f"the orchestrator cannot tell which release created its workers, so it cannot select their pods: "
        f"set {RELEASE_ENV_VAR}"
    )
    return release
