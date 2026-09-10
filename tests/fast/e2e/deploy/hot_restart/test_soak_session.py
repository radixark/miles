from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import REPLACED_LAUNCH_EXIT_CODE
from tests.e2e.deploy.conftest_deploy.hot_restart.soak_session import assert_hot_restart_launches_finished
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakDeploymentTarget,
    SoakLauncherExitedEvent,
)


@pytest.mark.parametrize("final_code", [0, REPLACED_LAUNCH_EXIT_CODE, 1])
def test_only_an_applied_successor_can_explain_a_replaced_launcher(final_code: int) -> None:
    """A replacement may explain the old launcher's exit but cannot excuse the final launcher's failure."""
    request = SoakActionRequest(
        target=SoakDeploymentTarget(
            namespace="rl",
            release="demo",
            workload_stamps={},
            workload_uids={},
            saved_iteration=2,
            finished_rollout_id=3,
        ),
        form_name="hot_restart",
        harms_cell=False,
    )
    events = [
        SoakActionRequestedEvent(request=request),
        SoakLauncherExitedEvent(request_id=None, returncode=REPLACED_LAUNCH_EXIT_CODE, log_path=Path("initial.log")),
        SoakActionAppliedEvent(request_id=request.request_id, evidence={}),
        SoakLauncherExitedEvent(request_id=request.request_id, returncode=final_code, log_path=Path("restart.log")),
    ]
    if final_code:
        with pytest.raises(AssertionError, match="final launcher|Launcher failed"):
            assert_hot_restart_launches_finished(events)
    else:
        assert_hot_restart_launches_finished(events)

    with pytest.raises(AssertionError, match="No applied successor"):
        assert_hot_restart_launches_finished(
            [event for event in events if not isinstance(event, SoakActionAppliedEvent)]
        )
