import asyncio

from tests.utils.soak.deploy.soak_form import SESSION_TIMEOUT_SECONDS
from tests.utils.soak.deploy.utils import REPLACED_LAUNCH_EXIT_CODE
from tests.utils.soak.recipes.gsm8k import Gsm8kRun
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec, launch
from tests.utils.soak.runner import SoakRunner
from tests.utils.soak.state import SoakDeploymentTarget, SoakEvent, SoakLauncherExitedEvent
from tests.utils.soak.views import project_actions


def assert_hot_restart_launches_finished(events: list[SoakEvent]) -> None:
    actions = [
        action
        for action in project_actions(events).values()
        if isinstance(action.requested.request.target, SoakDeploymentTarget)
    ]
    exits = [event for event in events if isinstance(event, SoakLauncherExitedEvent)]
    expected = [None, *(action.requested.request.request_id for action in actions)]
    assert len(exits) == len(expected), f"Missing or repeated launcher exits: expected {expected}, got {exits}"
    for index, request_id in enumerate(expected):
        matching = [event for event in exits if event.request_id == request_id]
        assert len(matching) == 1, f"Expected one launcher exit for {request_id}: {matching}"
        [exit_event] = matching
        if exit_event.returncode == REPLACED_LAUNCH_EXIT_CODE:
            assert index < len(actions), f"The final launcher was replaced with no successor: {exit_event}"
            successor = actions[index]
            assert (
                successor.applied is not None and successor.requested.timestamp <= exit_event.timestamp
            ), f"No applied successor explains the replacement exit: {exit_event}"
        else:
            assert exit_event.returncode == 0, f"Launcher failed: {exit_event}"


async def execute_hot_restart_session(run: Gsm8kRun, injector: SoakRunner) -> None:
    async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
        async with asyncio.TaskGroup() as tasks:
            initial = tasks.create_task(_launch_initial(run))
            try:
                await _wait_for_training(run=run)
            finally:
                if not initial.done():
                    initial.cancel()
    assert_hot_restart_launches_finished(run.event_log.events)


async def _launch_initial(run: Gsm8kRun) -> None:
    log_path = run.evidence_dir / "launcher-initial.log"
    result = await launch(
        Gsm8kLaunchSpec(config=run.config, train_args=run.train_args),
        log_path=log_path,
        timeout_seconds=SESSION_TIMEOUT_SECONDS,
    )
    run.event_log.note_launcher_exited(SoakLauncherExitedEvent(request_id=None, returncode=result, log_path=log_path))
    assert result in (0, REPLACED_LAUNCH_EXIT_CODE), f"Initial launcher exited {result}; see {log_path}"


async def _wait_for_training(*, run: Gsm8kRun) -> None:
    while True:
        events = run.event_log.events
        actions = {
            request_id: action
            for request_id, action in project_actions(events).items()
            if isinstance(action.requested.request.target, SoakDeploymentTarget)
        }
        failed = [
            action.result for action in actions.values() if action.result is not None and not action.result.returned
        ]
        assert not failed, f"Hot restart actions failed: {failed}"
        exits = {event.request_id: event for event in events if isinstance(event, SoakLauncherExitedEvent)}
        if (initial := exits.get(None)) is not None:
            if not actions:
                assert initial.returncode == 0, f"The initial launcher was replaced before any request: {initial}"
                return
            if all(action.result is not None for action in actions.values()):
                latest = exits.get(next(reversed(actions)))
                if latest is not None and latest.returncode == 0:
                    return
        await asyncio.sleep(0.2)
