"""E2E negative test: no false positives during normal training.

Parametrized across TRAINING_FOCUSED and ROLLOUT_FOCUSED topologies to verify
that both training and rollout subsystems remain stable without faults.
"""

from __future__ import annotations

import pytest
import ray
from tests.e2e.ft.conftest import ROLLOUT_FOCUSED, TRAINING_FOCUSED, get_status
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_no_false_positive


@pytest.mark.parametrize(
    "ft_controller_handle",
    [TRAINING_FOCUSED, ROLLOUT_FOCUSED],
    indirect=True,
)
async def test_no_false_positive_during_normal_training(
    ft_controller_handle: ray.actor.ActorHandle,
) -> None:
    """Controller should not trigger recovery when training runs normally."""
    status = await scenario_no_false_positive(
        handle=ft_controller_handle,
        observation_iterations=10,
        timeout=120.0,
        poll_interval=5.0,
    )

    # Step 1: all subsystems should be in DetectingAnomalySt
    assert status.subsystem_states, "Expected non-empty subsystem_states"
    for name, state in status.subsystem_states.items():
        assert state == "DetectingAnomalySt", (
            f"Subsystem {name} in unexpected state {state}, expected DetectingAnomalySt"
        )
