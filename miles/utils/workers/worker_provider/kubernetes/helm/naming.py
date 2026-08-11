from __future__ import annotations

import hashlib

from miles.utils.workers.worker_spec import BaseWorkerSpec, HostAndPort, NamedHostAndPorts

CHART_NAME = "miles-run"
UNINSTALLER_SERVICE_ACCOUNT = "miles-uninstaller"

MAX_OBJECT_NAME_LENGTH = 63
LONGEST_CELL_INDEX_SUFFIX = "-999"
LONGEST_REVISION_HASH_SUFFIX = "-0123456789"
COMPONENT_NAME_BUDGET = MAX_OBJECT_NAME_LENGTH - len(LONGEST_CELL_INDEX_SUFFIX) - len(LONGEST_REVISION_HASH_SUFFIX)

RELEASE_DIGEST_LENGTH = 6


def static_cell_addrs(
    *, spec: BaseWorkerSpec, release: str, cell_index: int, worker_in_pod_index: int = 0
) -> NamedHostAndPorts:
    host = static_worker_host(release, spec.name, cell_index)
    return {
        port.name: HostAndPort(
            host=host, port=port.static_port + (worker_in_pod_index if port.mode == "per_worker" else 0)
        )
        for port in spec.port_infos
    }


def static_worker_host(release: str, component: str, cell_index: int) -> str:
    name = component_name(release, component)
    return f"{name}-{cell_index}.{name}"


def component_name(release: str, component: str) -> str:
    budget = COMPONENT_NAME_BUDGET - (len(component) + 1)
    return f"{release_prefix(release, chart_name=CHART_NAME, budget=budget)}-{component}"


def release_prefix(release: str, *, chart_name: str, budget: int) -> str:
    name = release if chart_name in release else f"{release}-{chart_name}"
    if len(name) <= budget:
        return name
    digest = hashlib.blake2b(release.encode(), digest_size=RELEASE_DIGEST_LENGTH).hexdigest()
    return f"{_trim_suffix(name[: budget - (len(digest) + 1)], '-')}-{digest}"


def _trim_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value
