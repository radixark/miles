from __future__ import annotations

from miles.utils.workers.worker_spec import BaseWorkerSpec, HostAndPort, NamedHostAndPorts

CHART_NAME = "miles-run"

MAX_OBJECT_NAME_LENGTH = 63
LONGEST_CELL_INDEX_SUFFIX = "-999"
LONGEST_REVISION_HASH_SUFFIX = "-0123456789"
COMPONENT_NAME_BUDGET = MAX_OBJECT_NAME_LENGTH - len(LONGEST_CELL_INDEX_SUFFIX) - len(LONGEST_REVISION_HASH_SUFFIX)


def static_cell_addrs(*, spec: BaseWorkerSpec, release: str, cell_index: int) -> NamedHostAndPorts:
    host = static_worker_host(release, spec.name, cell_index)
    return {port.name: HostAndPort(host=host, port=port.static_port) for port in spec.port_infos}


def static_worker_host(release: str, component: str, cell_index: int) -> str:
    name = component_name(release, component)
    return f"{name}-{cell_index}.{name}"


def component_name(release: str, component: str) -> str:
    budget = COMPONENT_NAME_BUDGET - (len(component) + 1)
    prefix = _trim_suffix(_release_prefix(release)[:budget], "-")
    return f"{prefix}-{component}"


def _release_prefix(release: str) -> str:
    name = release if CHART_NAME in release else f"{release}-{CHART_NAME}"
    return _trim_suffix(name[:COMPONENT_NAME_BUDGET], "-")


def _trim_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value
