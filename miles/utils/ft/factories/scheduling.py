"""Scheduling helpers for placing FT actors on non-GPU nodes."""

from __future__ import annotations

from typing import Any

from miles.utils.ft.adapters.impl.ray.node_discovery import CPU_ONLY_RESOURCE, assert_cpu_only_nodes_exist


def get_cpu_only_scheduling_options() -> dict[str, Any]:
    """Return Ray actor ``.options()`` kwargs that schedule onto a cpu-only node.

    Validates that the cluster actually has at least one node with the
    ``cpu_only`` custom resource before returning, so callers fail fast
    instead of blocking on an unsatisfiable resource request.
    """
    assert_cpu_only_nodes_exist()
    return {"resources": {CPU_ONLY_RESOURCE: 0.001}}
