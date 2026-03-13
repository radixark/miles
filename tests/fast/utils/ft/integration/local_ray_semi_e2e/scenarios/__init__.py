"""Shared scenario functions for E2E and semi-E2E FT tests.

Previously E2E tests imported scenario functions from this module but the
functions were never implemented (ModuleNotFoundError at runtime). Semi-E2E
tests had the same logic inlined with zero code sharing.

These scenario functions accept protocol objects (FaultTestProtocol,
FaultInjectionProtocol) so the same scenario runs against both MilesTestbed
(semi-E2E) and real Ray clusters (E2E).
"""

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.hang import scenario_hang_detection_and_recovery
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.multi_cell_crash import scenario_multi_cell_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.no_false_positive import scenario_no_false_positive
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultInjectionProtocol,
    FaultTestProtocol,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.repeated_crash import scenario_repeated_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.rollout_crash import scenario_rollout_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.rollout_gpu_xid import scenario_rollout_gpu_xid
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.rollout_repeated_crash import (
    scenario_rollout_repeated_crash,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.transient_crash import scenario_transient_crash

__all__ = [
    "FaultInjectionProtocol",
    "FaultTestProtocol",
    "scenario_hang_detection_and_recovery",
    "scenario_multi_cell_crash",
    "scenario_no_false_positive",
    "scenario_repeated_crash",
    "scenario_rollout_crash",
    "scenario_rollout_gpu_xid",
    "scenario_rollout_repeated_crash",
    "scenario_transient_crash",
]
