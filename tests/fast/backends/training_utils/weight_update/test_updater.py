from argparse import Namespace
from typing import Any

import pytest

from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

_GLOO_GROUP = object()


class _GatheredVerdicts:
    def __init__(self, verdicts_of_other_ranks: list[list[str]]) -> None:
        self.contributed: list[list[str]] = []
        self._verdicts_of_other_ranks = verdicts_of_other_ranks

    @property
    def world_size(self) -> int:
        return len(self._verdicts_of_other_ranks) + 1

    def all_gather_object(self, object_list: list, obj: Any, group: Any) -> None:
        assert group is _GLOO_GROUP
        self.contributed.append(obj)
        object_list[0] = obj
        for index, verdict in enumerate(self._verdicts_of_other_ranks):
            object_list[index + 1] = verdict


def _install_fake_gloo(monkeypatch: pytest.MonkeyPatch, gathered: _GatheredVerdicts) -> None:
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: _GLOO_GROUP)
    monkeypatch.setattr(updater_module.dist, "get_world_size", lambda group: gathered.world_size)
    monkeypatch.setattr(updater_module.dist, "all_gather_object", gathered.all_gather_object)


def _make_updaters(cell_ids: list[str]) -> list[_RolloutCellUpdater]:
    return [
        _RolloutCellUpdater(
            args=Namespace(update_weight_engine_request_timeout=10.0), cell_id=cell_id, api_client=object()
        )
        for cell_id in cell_ids
    ]


class TestMarkCellsErroredOnAnyRank:
    def test_a_cell_another_rank_could_not_reach_is_given_up_here_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An engine holding a half-written model must not be resumed just because this rank reached it."""
        updaters = _make_updaters(["cell-a", "cell-b"])
        _install_fake_gloo(monkeypatch, _GatheredVerdicts([["cell-b"]]))

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert not updaters[0].is_errored
        assert updaters[1].is_errored
        assert "another trainer rank" in str(updaters[1]._error)

    def test_a_cell_every_rank_reached_stays_updatable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Marking a healthy cell would drop an engine out of service for no reason."""
        updaters = _make_updaters(["cell-a", "cell-b"])
        _install_fake_gloo(monkeypatch, _GatheredVerdicts([[], []]))

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert not any(u.is_errored for u in updaters)

    def test_this_rank_contributes_exactly_the_cells_it_lost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other ranks can only skip a cell this rank lost if this rank reports it."""
        updaters = _make_updaters(["cell-a", "cell-b", "cell-c"])
        updaters[2].mark_errored(RuntimeError("write failed"))
        gathered = _GatheredVerdicts([[]])
        _install_fake_gloo(monkeypatch, gathered)

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert gathered.contributed == [["cell-c"]]

    def test_one_rank_losing_a_cell_is_enough_to_give_it_up_everywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The verdict is a union, not a vote, because a partial write is enough to break the engine."""
        updaters = _make_updaters(["cell-a", "cell-b", "cell-c"])
        _install_fake_gloo(monkeypatch, _GatheredVerdicts([[], ["cell-a"], []]))

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert updaters[0].is_errored
        assert not updaters[1].is_errored
        assert not updaters[2].is_errored

    def test_a_cell_that_already_failed_here_keeps_its_original_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overwriting the local cause with the aggregated verdict would hide why the cell was lost."""
        updaters = _make_updaters(["cell-a"])
        local_error = RuntimeError("rdma write failed")
        updaters[0].mark_errored(local_error)
        _install_fake_gloo(monkeypatch, _GatheredVerdicts([["cell-a"]]))

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert updaters[0]._error is local_error

    def test_a_cell_id_this_rank_does_not_hold_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ranks talk to different subsets of cells, so an unknown id must not raise."""
        updaters = _make_updaters(["cell-a"])
        _install_fake_gloo(monkeypatch, _GatheredVerdicts([["cell-z"]]))

        updater_module._mark_cells_errored_on_any_rank(updaters)

        assert not updaters[0].is_errored

    def test_an_empty_cell_list_still_takes_part_in_the_gather(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rank that skips the collective would hang every other trainer rank."""
        gathered = _GatheredVerdicts([["cell-a"]])
        _install_fake_gloo(monkeypatch, gathered)

        updater_module._mark_cells_errored_on_any_rank([])

        assert gathered.contributed == [[]]
