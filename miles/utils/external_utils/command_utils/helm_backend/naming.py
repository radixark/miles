from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.naming import (
    CHART_NAME,
    NAME_BUDGET,
    ORCHESTRATOR_COMPONENT,
    RUN_ID_PATTERN,
    cell_leader_host,
    component_name,
    fullname,
    orchestrator_host,
    release_name,
    static_worker_host,
)

__all__ = [
    "CHART_NAME",
    "NAME_BUDGET",
    "ORCHESTRATOR_COMPONENT",
    "RUN_ID_PATTERN",
    "component_name",
    "cell_leader_host",
    "fullname",
    "orchestrator_host",
    "release_name",
    "static_worker_host",
]
