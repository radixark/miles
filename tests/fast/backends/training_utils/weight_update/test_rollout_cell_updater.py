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
