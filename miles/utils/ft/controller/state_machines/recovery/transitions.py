"""Recovery transition policy: state construction separated from data models.

The EvictingAndRestartingSt model previously embedded flow-strategy
classmethods (direct_restart, evict_and_restart_next_stop_time_diag,
evict_and_restart_final).  Those methods coupled transition logic to the
data model, meaning that adding a new recovery phase required changes
in both the model file and the handler file.

This module centralises all "given this trigger / diagnostic result,
which EvictingAndRestartingSt should be produced?" decisions so that
models.py stays a pure state data definition.
"""

from __future__ import annotations

from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    StoppingAndRestartingSt,
)


def direct_restart() -> EvictingAndRestartingSt:
    """No identified bad nodes: restart directly, diagnose on failure."""
    return EvictingAndRestartingSt(
        restart=StoppingAndRestartingSt(),
        failed_next_state=StopTimeDiagnosticsSt(),
    )


def evict_and_restart_next_stop_time_diag(
    *,
    bad_node_ids: tuple[str, ...],
) -> EvictingAndRestartingSt:
    """Evict known bad nodes and restart; diagnose on failure."""
    return EvictingAndRestartingSt(
        restart=EvictingSt(bad_node_ids=bad_node_ids),
        failed_next_state=StopTimeDiagnosticsSt(),
    )


def evict_and_restart_final(
    *,
    bad_node_ids: tuple[str, ...],
) -> EvictingAndRestartingSt:
    """Final retry after diagnostics: evict and restart; notify humans on failure."""
    return EvictingAndRestartingSt(
        restart=EvictingSt(bad_node_ids=bad_node_ids),
        failed_next_state=NotifyHumansSt(
            state_before="EvictingAndRestartingSt",
            reason="final_restart_failed",
        ),
    )
