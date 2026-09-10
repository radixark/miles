import asyncio
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import soak_form
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_release_of_config
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec
from tests.utils.soak.state import (
    EventLog,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakDeploymentTarget,
    SoakLauncherExitedEvent,
    SoakObservation,
)
from tests.utils.soak.views import compute_successful_form_names

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


@pytest.mark.parametrize("exit_code", [0, 1])
def test_applied_take_over_is_visible_before_launcher_finishes_and_late_failure_is_not_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exit_code: int
) -> None:
    """Record applied replacement before training finishes, while preserving the launcher's final verdict."""

    async def scenario() -> None:
        entered = asyncio.Event()
        finish = asyncio.Event()
        config = ExecuteTrainConfig(run_id="demo", namespace="rl", cluster_backend=ClusterBackend.KUBERNETES)
        release = compute_release_of_config(config)
        workloads = compute_hot_restart_workloads(release)
        target = SoakDeploymentTarget(
            namespace=config.namespace,
            release=release,
            workload_stamps={name: "before" for name in workloads},
            workload_uids={name: f"uid-{name}" for name in workloads},
            saved_iteration=2,
            finished_rollout_id=3,
        )
        request = SoakActionRequest(target=target, form_name="hot_restart", harms_cell=False)
        restored = SoakActionRequest.model_validate_json(request.model_dump_json())
        assert restored == request and isinstance(restored.target, SoakDeploymentTarget)
        log = EventLog()
        log.note_action_requested(request)

        async def launch(spec: Gsm8kLaunchSpec, *, log_path: Path, timeout_seconds: float) -> int:
            assert spec.config.run_id == config.run_id
            assert spec.config.namespace == config.namespace
            assert spec.config.hot_restart
            entered.set()
            await finish.wait()
            return exit_code

        monkeypatch.setattr(soak_form, "launch", launch)
        form = soak_form.SoakActionFormHotRestart(
            launch_spec=Gsm8kLaunchSpec(config=config, train_args="--save-interval 3", fully_async=False),
            event_log=log,
            log_dir=tmp_path,
            max_allowed_rollout_id=234,
            poll_interval_seconds=0,
        )
        task = asyncio.create_task(form.execute(request))
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                log.note_observation(
                    SoakObservation(
                        cells=[],
                        deployments=[
                            target.model_copy(update={"workload_stamps": {name: "after" for name in workloads}})
                        ],
                    )
                )
                while not any(isinstance(event, SoakActionAppliedEvent) for event in log.events):
                    await asyncio.sleep(0)
                assert not task.done()
                assert not any(isinstance(event, SoakLauncherExitedEvent) for event in log.events)
                assert compute_successful_form_names(log.events, cell_type="deployment") == {"hot_restart"}
                finish.set()
                if exit_code:
                    with pytest.raises(AssertionError, match="exited 1"):
                        await task
                else:
                    await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        exits = [event for event in log.events if isinstance(event, SoakLauncherExitedEvent)]
        assert len(exits) == 1
        assert exits[0].request_id == request.request_id and exits[0].returncode == exit_code

    asyncio.run(scenario())
