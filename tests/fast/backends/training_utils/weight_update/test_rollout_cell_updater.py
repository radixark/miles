from concurrent.futures import Future
from typing import Any

import pytest

from miles.backends.training_utils.weight_update.rollout_cell_updater import (
    _RolloutCellUpdater,
    create_rollout_cell_updaters,
)


class _FakeApiClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, dict]] = []

    async def get_server_info(self, **kwargs: Any) -> str:
        self.calls.append(("get_server_info", kwargs))
        return f"info-{self.name}"


class TestSubmitClientCall:
    """Every engine request of a weight update goes through the updater of its own cell."""

    def test_a_client_call_reaches_the_engine_and_hands_back_its_answer(self) -> None:
        """The caller needs the engine's answer, so the future must carry it rather than only its completion."""
        client = _FakeApiClient("a")
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=client)

        future: Future = updater.submit_client_call("get_server_info")

        assert future.result(timeout=10) == "info-a"
        assert client.calls == [("get_server_info", {})]

    def test_the_call_arguments_reach_the_engine_untouched(self) -> None:
        """A dropped argument would silently change what the engine is asked to do."""
        client = _FakeApiClient("a")
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=client)

        updater.submit_client_call("get_server_info", rank=3).result(timeout=10)

        assert client.calls == [("get_server_info", {"rank": 3})]

    def test_an_updater_talks_only_to_the_engine_of_its_own_cell(self) -> None:
        """One cell's request landing on another cell's engine pauses or resumes the wrong sampler."""
        first, second = _FakeApiClient("first"), _FakeApiClient("second")
        updater = _RolloutCellUpdater(cell_id="cell-first", api_client=first)

        updater.submit_client_call("get_server_info").result(timeout=10)

        assert second.calls == []


class TestCreateRolloutCellUpdaters:
    """The mapping from cell id to updater the whole weight update is driven through."""

    def test_every_rollout_engine_gets_an_updater_keyed_by_its_cell_id(self) -> None:
        """The trainer reports failures per cell id, so the lookup has to be keyed by it."""
        clients = [_FakeApiClient("a"), _FakeApiClient("b")]

        updaters = create_rollout_cell_updaters(clients, ["cell-a", "cell-b"])

        assert sorted(updaters) == ["cell-a", "cell-b"]
        assert updaters["cell-a"].cell_id == "cell-a"

    def test_each_updater_is_paired_with_the_engine_at_the_same_position(self) -> None:
        """A shifted pairing sends every cell's session calls to its neighbour's engine."""
        first, second = _FakeApiClient("a"), _FakeApiClient("b")

        updaters = create_rollout_cell_updaters([first, second], ["cell-a", "cell-b"])
        updaters["cell-b"].submit_client_call("get_server_info").result(timeout=10)

        assert (first.calls, [name for name, _ in second.calls]) == ([], ["get_server_info"])

    def test_metadata_that_describes_a_different_set_of_engines_is_rejected(self) -> None:
        """Fewer cell ids than engines would leave an engine nobody can name, blame or pause."""
        with pytest.raises(ValueError):
            create_rollout_cell_updaters([_FakeApiClient("a"), _FakeApiClient("b")], ["cell-a"])


class _FailingApiClient(_FakeApiClient):
    async def get_server_info(self, **kwargs: Any) -> str:
        self.calls.append(("get_server_info", kwargs))
        raise RuntimeError(f"{self.name} is gone")


class TestErroredCellIsLeftAlone:
    """A cell that failed its update must not be talked to again by this weight update."""

    def test_a_fresh_cell_is_not_errored(self) -> None:
        """Every cell starts updatable, or the first weight update would reach nobody."""
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=_FakeApiClient("a"))

        assert updater.is_errored is False

    def test_a_failing_request_marks_the_cell_instead_of_raising_for_the_fleet(self) -> None:
        """One unreachable sampler must not abort the weight update of every other sampler."""
        client = _FailingApiClient("a")
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=client)

        assert updater.submit_client_call("get_server_info").result(timeout=10) is None
        assert updater.is_errored is True

    def test_an_errored_cell_never_reaches_its_engine_again(self) -> None:
        """Later session calls to a cell that already failed would only waste the trainer's deadline."""
        client = _FailingApiClient("a")
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=client)
        updater.submit_client_call("get_server_info").result(timeout=10)

        assert updater.submit_client_call("get_server_info").result(timeout=10) is None
        assert len(client.calls) == 1

    def test_the_first_error_is_the_one_that_is_kept(self) -> None:
        """The first failure is the one that explains why the cell lost the update."""
        updater = _RolloutCellUpdater(cell_id="cell-a", api_client=_FakeApiClient("a"))
        first = RuntimeError("first")

        updater.mark_errored(first)
        updater.mark_errored(RuntimeError("second"))

        assert updater._error is first

    def test_marking_is_confined_to_the_cell_it_was_called_on(self) -> None:
        """Blaming a healthy cell would take a serving sampler out of the fleet for nothing."""
        first = _RolloutCellUpdater(cell_id="cell-a", api_client=_FakeApiClient("a"))
        second = _RolloutCellUpdater(cell_id="cell-b", api_client=_FakeApiClient("b"))

        first.mark_errored(RuntimeError("lost"))

        assert (first.is_errored, second.is_errored) == (True, False)
