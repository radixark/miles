"""Shared test scenario functions for local_ray and E2E reuse.

Each scenario takes a controller handle, a fault injector, and optional
parameters. The scenario drives the test through a standard sequence:
wait for stable training → inject fault → observe recovery → verify outcome.

Scenarios are agnostic to whether faults are injected by killing real
processes (E2E) or by changing simulated state (local_ray).
"""

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.hang_detection import (
    scenario_hang_detection,
    scenario_hang_detection_and_recovery,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.multi_cell_crash import scenario_multi_cell_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.no_false_positive import scenario_no_false_positive
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.polling import (
    get_status,
    wait_for_all_subsystems_detecting,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_subsystem_state,
    wait_for_training_stable,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.repeated_crash import scenario_repeated_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.rollout_crash import scenario_rollout_crash
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.transient_crash import scenario_transient_crash

__all__ = [
    "get_status",
    "scenario_hang_detection",
    "scenario_hang_detection_and_recovery",
    "scenario_multi_cell_crash",
    "scenario_no_false_positive",
    "scenario_repeated_crash",
    "scenario_rollout_crash",
    "scenario_transient_crash",
    "wait_for_all_subsystems_detecting",
    "wait_for_mode",
    "wait_for_mode_transition",
    "wait_for_recovery_complete",
    "wait_for_recovery_phase",
    "wait_for_subsystem_state",
    "wait_for_training_stable",
]
