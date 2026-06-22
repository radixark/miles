import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.ft_test_actions import _ACTOR_ACTIONS, _GROUP_ACTIONS, FTTestAction, _load_actions


def _args(ci_ft_test_actions: object) -> SimpleNamespace:
    return SimpleNamespace(ci_ft_test_actions=ci_ft_test_actions)


def test_load_actions_returns_empty_when_attr_is_none() -> None:
    """None ci_ft_test_actions yields an empty action list without parsing."""
    assert _load_actions(_args(None), _GROUP_ACTIONS) == []


def test_load_actions_returns_empty_when_attr_is_empty_string() -> None:
    """Empty-string ci_ft_test_actions is falsy and yields an empty list."""
    assert _load_actions(_args(""), _ACTOR_ACTIONS) == []


def test_load_actions_returns_empty_when_attr_missing() -> None:
    """A missing ci_ft_test_actions attribute defaults to None and yields []."""
    assert _load_actions(SimpleNamespace(), _GROUP_ACTIONS) == []


def test_load_actions_parses_single_crash_action_with_defaults() -> None:
    """A single crash_before_allreduce action loads with the model's default fields."""
    raw = json.dumps([{"at_rollout": 3, "action": "crash_before_allreduce"}])
    actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, FTTestAction)
    assert action.at_rollout == 3
    assert action.action == "crash_before_allreduce"
    assert action.cell_index == -1
    assert action.rank == 0
    assert action.attempt == 0


def test_load_actions_filters_to_only_matching_actions() -> None:
    """Mixed actions are filtered down to those whose action is in the filter set."""
    raw = json.dumps(
        [
            {"at_rollout": 1, "action": "stop_cell_at_end"},
            {"at_rollout": 2, "action": "crash_before_allreduce"},
            {"at_rollout": 3, "action": "start_cell_at_end"},
        ]
    )
    group_actions = _load_actions(_args(raw), _GROUP_ACTIONS)
    assert [a.action for a in group_actions] == ["stop_cell_at_end", "start_cell_at_end"]
    actor_actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert [a.action for a in actor_actions] == ["crash_before_allreduce"]


def test_load_actions_returns_empty_when_no_action_matches_filter() -> None:
    """Valid actions that fall outside the filter set produce an empty result."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce"}])
    assert _load_actions(_args(raw), _GROUP_ACTIONS) == []


def test_load_actions_rejects_extra_field() -> None:
    """An unexpected JSON field is rejected because the model forbids extras."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "bogus": 5}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _GROUP_ACTIONS)


def test_load_actions_rejects_invalid_action_literal() -> None:
    """An action string outside the allowed Literal set raises a validation error."""
    raw = json.dumps([{"at_rollout": 1, "action": "not_a_real_action"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _GROUP_ACTIONS)


def test_resolve_cell_index_uses_last_cell_for_default() -> None:
    """A default cell_index of -1 resolves to the last cell index (num_cells - 1)."""
    action = FTTestAction(at_rollout=0, action="stop_cell_at_end")
    assert action.resolve_cell_index(num_cells=4) == 3


def test_resolve_cell_index_keeps_explicit_index() -> None:
    """A non-negative cell_index is returned unchanged by resolve_cell_index."""
    action = FTTestAction(at_rollout=0, action="stop_cell_at_end", cell_index=1)
    assert action.resolve_cell_index(num_cells=4) == 1


from miles.utils.test_utils.ft_test_actions import FTTestActionGroupExecutor


class FakeGroup:
    def __init__(self, num_cells: int) -> None:
        self.num_cells = num_cells
        self.stopped: list[int] = []
        self.started: list[int] = []

    def stop_cell(self, cell_index: int) -> None:
        self.stopped.append(cell_index)

    def start_cell(self, cell_index: int) -> None:
        self.started.append(cell_index)


class TestResolveCellIndex:
    def test_non_negative_index_returned_as_is(self):
        """resolve_cell_index returns the explicit index when it is non-negative."""
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        assert action.resolve_cell_index(num_cells=3) == 1

    def test_negative_index_resolves_to_last_cell(self):
        """resolve_cell_index maps the default -1 to the last cell (num_cells - 1)."""
        action = FTTestAction(at_rollout=5, action="start_cell_at_end", cell_index=-1)
        assert action.resolve_cell_index(num_cells=3) == 2


class TestRunAfterStep:
    def test_stop_cell_fires_on_matching_rollout(self):
        """stop_cell_at_end triggers group.stop_cell with the resolved cell index on its rollout."""
        group = FakeGroup(num_cells=3)
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        executor = FTTestActionGroupExecutor(actions=[action], group=group)

        executor.run_after_step(5)

        assert group.stopped == [1]
        assert group.started == []

    def test_no_action_on_non_matching_rollout(self):
        """run_after_step does nothing when no action's at_rollout matches the given rollout."""
        group = FakeGroup(num_cells=3)
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        executor = FTTestActionGroupExecutor(actions=[action], group=group)

        executor.run_after_step(4)

        assert group.stopped == []
        assert group.started == []

    def test_start_cell_with_default_index_resolves_to_last_cell(self):
        """start_cell_at_end with cell_index -1 calls group.start_cell on the last cell."""
        group = FakeGroup(num_cells=3)
        action = FTTestAction(at_rollout=2, action="start_cell_at_end", cell_index=-1)
        executor = FTTestActionGroupExecutor(actions=[action], group=group)

        executor.run_after_step(2)

        assert group.started == [2]
        assert group.stopped == []

    def test_two_actions_same_rollout_both_fire(self):
        """Two actions sharing the same rollout both dispatch to their respective group methods."""
        group = FakeGroup(num_cells=3)
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_index=0)
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_index=2)
        executor = FTTestActionGroupExecutor(actions=[stop_action, start_action], group=group)

        executor.run_after_step(7)

        assert group.stopped == [0]
        assert group.started == [2]

    def test_empty_actions_is_noop(self):
        """An executor with no actions performs no group calls."""
        group = FakeGroup(num_cells=3)
        executor = FTTestActionGroupExecutor(actions=[], group=group)

        executor.run_after_step(5)

        assert group.stopped == []
        assert group.started == []
