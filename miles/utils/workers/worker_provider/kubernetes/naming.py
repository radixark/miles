from __future__ import annotations

import re

CHART_NAME = "miles-run"
NAME_BUDGET = 52
ORCHESTRATOR_COMPONENT = "orchestrator"
RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def release_name(run_id: str) -> str:
    return f"{CHART_NAME}-{run_id}"


def fullname(release: str, chart_name: str = CHART_NAME) -> str:
    name = release if chart_name in release else f"{release}-{chart_name}"
    return _trim_suffix(_trunc(name, NAME_BUDGET), "-")


def component_name(release: str, component: str, chart_name: str = CHART_NAME) -> str:
    budget = NAME_BUDGET - (len(component) + 1)
    prefix = _trim_suffix(_trunc(fullname(release, chart_name), budget), "-")
    return f"{prefix}-{component}"


def static_worker_host(release: str, component: str, cell_index: int = 0) -> str:
    name = component_name(release, component)
    return f"{name}-{cell_index}.{name}"


def cell_leader_host(release: str, component: str, cell_ordinal: int) -> str:
    name = component_name(release, component)
    return f"{name}-{cell_ordinal}.{name}"


def orchestrator_host(release: str, namespace: str) -> str:
    return f"{component_name(release, ORCHESTRATOR_COMPONENT)}.{namespace}.svc.cluster.local"


def _trunc(value: str, count: int) -> str:
    return value[count:] if count < 0 else value[:count]


def _trim_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value
