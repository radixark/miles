import asyncio

import pytest
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.deploy import session as session_module
from tests.utils.soak.deploy.session import LauncherChain, execute_hot_restart_session


def _outcome(outcome: LaunchOutcome) -> asyncio.Task[LaunchOutcome]:
    async def _return() -> LaunchOutcome:
        return outcome

    return asyncio.create_task(_return())


def _failure(error: BaseException) -> asyncio.Task[LaunchOutcome]:
    async def _raise() -> LaunchOutcome:
        raise error

    return asyncio.create_task(_raise())


def _install_first_launch(monkeypatch: pytest.MonkeyPatch, first: LaunchOutcome | BaseException) -> list[object]:
    runs: list[object] = []

    async def _execute_gsm8k_session(run: object, *, accept_replaced: bool) -> LaunchOutcome:
        runs.append(run)
        if isinstance(first, BaseException):
            raise first
        return first

    monkeypatch.setattr(session_module, "execute_gsm8k_session", _execute_gsm8k_session)
    return runs


class TestExecuteHotRestartSession:
    async def test_a_finished_first_launch_still_waits_for_the_take_over_launcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first launcher follows the run to its end, so a take-over started meanwhile must be awaited too."""
        runs = _install_first_launch(monkeypatch, LaunchOutcome.FINISHED)
        chain = LauncherChain()
        successor = _outcome(LaunchOutcome.FINISHED)
        chain.put_nowait(successor)

        await execute_hot_restart_session("run", chain=chain)

        assert runs == ["run"]
        assert chain.empty() and successor.done()

    async def test_each_replaced_launcher_hands_over_to_the_next_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The session follows every successor until the chain is drained and the last one finished."""
        _install_first_launch(monkeypatch, LaunchOutcome.REPLACED)
        chain = LauncherChain()
        chain.put_nowait(_outcome(LaunchOutcome.REPLACED))
        chain.put_nowait(_outcome(LaunchOutcome.FINISHED))
        chain.put_nowait(_outcome(LaunchOutcome.FINISHED))

        await execute_hot_restart_session("run", chain=chain)

        assert chain.empty()

    async def test_a_replaced_launcher_without_a_successor_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SIGTERM exit that no take-over explains is an unexplained kill, not a finished run."""
        _install_first_launch(monkeypatch, LaunchOutcome.REPLACED)

        with pytest.raises(AssertionError, match="no take-over launcher succeeded it"):
            await execute_hot_restart_session("run", chain=LauncherChain())

    async def test_a_chain_ending_in_replaced_without_a_successor_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The last successor being replaced too still needs one more launcher behind it."""
        _install_first_launch(monkeypatch, LaunchOutcome.REPLACED)
        chain = LauncherChain()
        chain.put_nowait(_outcome(LaunchOutcome.REPLACED))

        with pytest.raises(AssertionError, match="no take-over launcher succeeded it"):
            await execute_hot_restart_session("run", chain=chain)

    @pytest.mark.parametrize("failing", ["first", "successor"])
    async def test_a_failed_launcher_is_raised(self, monkeypatch: pytest.MonkeyPatch, failing: str) -> None:
        """Whichever launcher fails, the session surfaces its error instead of a verdict."""
        error = RuntimeError("helm upgrade failed")
        _install_first_launch(monkeypatch, error if failing == "first" else LaunchOutcome.REPLACED)
        chain = LauncherChain()
        chain.put_nowait(_failure(error) if failing == "successor" else _outcome(LaunchOutcome.FINISHED))

        with pytest.raises(RuntimeError, match="helm upgrade failed"):
            await execute_hot_restart_session("run", chain=chain)

        await asyncio.gather(*[chain.get_nowait() for _ in range(chain.qsize())], return_exceptions=True)
