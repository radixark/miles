from __future__ import annotations

import re

from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME, component_name, static_worker_host

ORCHESTRATOR_COMPONENT = "orchestrator"
RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")

__all__ = ["CHART_NAME", "ORCHESTRATOR_COMPONENT", "RUN_ID_PATTERN", "component_name", "static_worker_host"]


def release_name(run_id: str) -> str:
    return f"{CHART_NAME}-{run_id}"


def orchestrator_host(release: str, namespace: str) -> str:
    return f"{component_name(release, ORCHESTRATOR_COMPONENT)}.{namespace}.svc.cluster.local"
