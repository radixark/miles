import asyncio

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import REPLACED_LAUNCH_EXIT_CODE
from tests.e2e.deploy.conftest_deploy.hot_restart.soak_form import SESSION_TIMEOUT_SECONDS
from tests.utils.soak.entrypoint import FaultInjectorHandle
from tests.utils.soak.recipes.gsm8k import Gsm8kRun
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec, launch
from tests.utils.soak.state import (
    Event,
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakDeploymentTarget,
    SoakLauncherExitedEvent,
)


def execute_hot_restart_session(run: Gsm8kRun, injector: FaultInjectorHandle) -> None:
    asyncio.run(_run_session(run=run, injector=injector))


def assert_hot_restart_launches_finished(events: list[Event]) -> None:
    requests = [
        event
        for event in events
        if isinstance(event, SoakActionRequestedEvent) and isinstance(event.request.target, SoakDeploymentTarget)
    ]
    applied = {event.request_id for event in events if isinstance(event, SoakActionAppliedEvent)}
    exits = [event for event in events if isinstance(event, SoakLauncherExitedEvent)]
    expected = [None, *(event.request.request_id for event in requests)]
    assert len(exits) == len(expected), f"Missing or repeated launcher exits: expected {expected}, got {exits}"
    for index, request_id in enumerate(expected):
        matching = [event for event in exits if event.request_id == request_id]
        assert len(matching) == 1, f"Expected one launcher exit for {request_id}: {matching}"
        [exit_event] = matching
        if exit_event.returncode == REPLACED_LAUNCH_EXIT_CODE:
            assert index < len(requests), f"The final launcher was replaced with no successor: {exit_event}"
            successor = requests[index]
            assert (
                successor.request.request_id in applied and successor.timestamp <= exit_event.timestamp
            ), f"No applied successor explains the replacement exit: {exit_event}"
        else:
            assert exit_event.returncode == 0, f"Launcher failed: {exit_event}"


async def _run_session(*, run: Gsm8kRun, injector: FaultInjectorHandle) -> None:
    async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
        async with asyncio.TaskGroup() as tasks:
            initial = tasks.create_task(_launch_initial(run))
            try:
                await _wait_for_training(run=run, injector=injector)
            finally:
                if not initial.done():
                    initial.cancel()
    assert_hot_restart_launches_finished(run.event_log.events)


async def _launch_initial(run: Gsm8kRun) -> None:
    log_path = run.evidence_dir / "launcher-initial.log"
    result = await launch(
        Gsm8kLaunchSpec(config=run.config, train_args=run.train_args, fully_async=False),
        log_path=log_path,
        timeout_seconds=SESSION_TIMEOUT_SECONDS,
    )
    run.event_log.note_launcher_exited(SoakLauncherExitedEvent(request_id=None, returncode=result, log_path=log_path))
    assert result in (0, REPLACED_LAUNCH_EXIT_CODE), f"Initial launcher exited {result}; see {log_path}"


async def _wait_for_training(*, run: Gsm8kRun, injector: FaultInjectorHandle) -> None:
    while True:
        injector.raise_if_failed()
        events = run.event_log.events
        requests = [
            event.request.request_id
            for event in events
            if isinstance(event, SoakActionRequestedEvent) and isinstance(event.request.target, SoakDeploymentTarget)
        ]
        results = {event.request_id: event for event in events if isinstance(event, SoakActionResultEvent)}
        failed = [
            results[request_id]
            for request_id in requests
            if request_id in results and not results[request_id].returned
        ]
        assert not failed, f"Hot restart actions failed: {failed}"
        exits = {event.request_id: event for event in events if isinstance(event, SoakLauncherExitedEvent)}
        if (initial := exits.get(None)) is not None:
            if not requests:
                assert initial.returncode == 0, f"The initial launcher was replaced before any request: {initial}"
                return
            if all(request_id in results for request_id in requests):
                latest = exits.get(requests[-1])
                if latest is not None and latest.returncode == 0:
                    return
        await asyncio.sleep(0.2)
