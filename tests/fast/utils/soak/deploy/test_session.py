import asyncio
from pathlib import Path

import pytest
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.core.utils import REPLACED_LAUNCH_EXIT_CODE
from tests.utils.soak.deploy.session import LauncherChain, follow_launchers, start_launch

from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError


def _outcome(outcome: LaunchOutcome) -> asyncio.Task[LaunchOutcome]:
    async def _return() -> LaunchOutcome:
        return outcome

    return asyncio.create_task(_return())


def _failure(error: BaseException) -> asyncio.Task[LaunchOutcome]:
    async def _raise() -> LaunchOutcome:
        raise error

    return asyncio.create_task(_raise())


class TestFollowLaunchers:
    async def test_a_finished_first_launch_still_waits_for_every_launcher_left_on_the_chain(self) -> None:
        """A launcher that follows the same run has to return before the session ends."""
        chain = LauncherChain()
        successor = _outcome(LaunchOutcome.FINISHED)
        chain.put_nowait(successor)

        await follow_launchers(_outcome(LaunchOutcome.FINISHED), chain=chain)

        assert chain.empty() and successor.done()

    async def test_each_replaced_launcher_hands_over_to_the_next_in_order(self) -> None:
        """The session follows every successor until one of them finishes the run, then waits for the rest."""
        chain = LauncherChain()
        launchers = [
            _outcome(LaunchOutcome.REPLACED),
            _outcome(LaunchOutcome.FINISHED),
            _outcome(LaunchOutcome.FINISHED),
        ]
        for launcher in launchers:
            chain.put_nowait(launcher)

        await follow_launchers(_outcome(LaunchOutcome.REPLACED), chain=chain)

        assert chain.empty() and all(launcher.done() for launcher in launchers)

    async def test_a_replaced_launcher_without_a_successor_fails(self) -> None:
        """A SIGTERM exit that no take-over explains is an unexplained kill, not a finished run."""
        with pytest.raises(AssertionError, match="no take-over launcher succeeded it"):
            await follow_launchers(_outcome(LaunchOutcome.REPLACED), chain=LauncherChain())

    async def test_a_chain_ending_in_replaced_without_a_successor_fails(self) -> None:
        """The last successor being replaced too still needs one more launcher behind it."""
        chain = LauncherChain()
        chain.put_nowait(_outcome(LaunchOutcome.REPLACED))

        with pytest.raises(AssertionError, match="no take-over launcher succeeded it"):
            await follow_launchers(_outcome(LaunchOutcome.REPLACED), chain=chain)

    @pytest.mark.parametrize("failing", ["first", "successor"])
    async def test_a_failed_launcher_is_raised(self, failing: str) -> None:
        """Whichever launcher fails, the session surfaces its error instead of a verdict."""
        error = RuntimeError("helm upgrade failed")
        first = _failure(error) if failing == "first" else _outcome(LaunchOutcome.REPLACED)
        chain = LauncherChain()
        chain.put_nowait(_failure(error) if failing == "successor" else _outcome(LaunchOutcome.FINISHED))

        with pytest.raises(RuntimeError, match="helm upgrade failed"):
            await follow_launchers(first, chain=chain)

        await asyncio.gather(*[chain.get_nowait() for _ in range(chain.qsize())], return_exceptions=True)


class TestFinalLauncherOutcome:
    async def test_finished_then_replaced_is_rejected(self) -> None:
        """A finished predecessor cannot hide an unexplained final replacement."""
        chain = LauncherChain()
        chain.put_nowait(_outcome(LaunchOutcome.REPLACED))
        with pytest.raises(AssertionError, match="final launcher"):
            await follow_launchers(_outcome(LaunchOutcome.FINISHED), chain=chain)


class TestStartLaunch:
    async def test_expected_takeover_is_recorded_as_replaced(self, tmp_path: Path) -> None:
        """The launcher entrypoint accepts the designated replacement exit code."""
        event_log = EventLog(tmp_path / "events.jsonl")
        chain = LauncherChain()

        async def replaced() -> None:
            raise RunExitedError(REPLACED_LAUNCH_EXIT_CODE)

        launcher = start_launch(
            replaced(),
            event_log=event_log,
            request_id="launch-1",
            chain=chain,
        )
        assert await launcher == LaunchOutcome.REPLACED
        assert event_log.events[-1].outcome == LaunchOutcome.REPLACED
        assert chain.get_nowait() is launcher
