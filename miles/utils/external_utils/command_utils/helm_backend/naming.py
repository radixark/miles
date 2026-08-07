from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.naming import (
    CHART_NAME,
    NAME_BUDGET,
    cell_leader_host,
    component_name,
    fullname,
    release_name,
    static_worker_host,
)

__all__ = [
    "CHART_NAME",
    "NAME_BUDGET",
    "component_name",
    "cell_leader_host",
    "fullname",
    "release_name",
    "static_worker_host",
]
