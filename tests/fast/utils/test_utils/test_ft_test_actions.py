import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.ft_test_actions import _ACTOR_ACTIONS, _CONTROLLER_ACTIONS, FTTestAction, _load_actions


def _args(ci_ft_test_actions: object) -> SimpleNamespace:
    return SimpleNamespace(ci_ft_test_actions=ci_ft_test_actions)


def test_load_actions_returns_empty_when_attr_is_none() -> None:
    """None ci_ft_test_actions yields an empty action list without parsing."""
    assert _load_actions(_args(None), _CONTROLLER_ACTIONS) == []


def test_load_actions_returns_empty_when_attr_is_empty_string() -> None:
    """Empty-string ci_ft_test_actions is falsy and yields an empty list."""
    assert _load_actions(_args(""), _ACTOR_ACTIONS) == []


def test_load_actions_returns_empty_when_attr_missing() -> None:
    """A missing ci_ft_test_actions attribute defaults to None and yields []."""
    assert _load_actions(SimpleNamespace(), _CONTROLLER_ACTIONS) == []


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
    group_actions = _load_actions(_args(raw), _CONTROLLER_ACTIONS)
    assert [a.action for a in group_actions] == ["stop_cell_at_end", "start_cell_at_end"]
    actor_actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert [a.action for a in actor_actions] == ["crash_before_allreduce"]


def test_load_actions_returns_empty_when_no_action_matches_filter() -> None:
    """Valid actions that fall outside the filter set produce an empty result."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce"}])
    assert _load_actions(_args(raw), _CONTROLLER_ACTIONS) == []


def test_load_actions_rejects_extra_field() -> None:
    """An unexpected JSON field is rejected because the model forbids extras."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "bogus": 5}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_invalid_action_literal() -> None:
    """An action string outside the allowed Literal set raises a validation error."""
    raw = json.dumps([{"at_rollout": 1, "action": "not_a_real_action"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


from miles.utils.test_utils.ft_test_actions import FTTestActionActorExecutor, FTTestActionControllerExecutor

_CELL_IDS = ["trainer-actor-0", "trainer-actor-1", "trainer-actor-2"]


class FakeController:
    def __init__(self, num_cells: int) -> None:
        self.cell_ids = _CELL_IDS[:num_cells]


class FakeRemoteMethod:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def remote(self, cell_ids: list[str]) -> None:
        self._sink.extend(cell_ids)


class FakeWorkerManager:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.started: list[str] = []
        self.stop_cells = FakeRemoteMethod(self.stopped)
        self.start_cells = FakeRemoteMethod(self.started)


class TestResolveCellId:
    def test_non_negative_index_selects_that_cell(self):
        """resolve_cell_id indexes the controller's cell ids when the index is explicit."""
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        assert action.resolve_cell_id(_CELL_IDS) == "trainer-actor-1"

    def test_negative_index_resolves_to_last_cell(self):
        """resolve_cell_id maps the default -1 to the last cell."""
        action = FTTestAction(at_rollout=5, action="start_cell_at_end", cell_index=-1)
        assert action.resolve_cell_id(_CELL_IDS) == "trainer-actor-2"

    def test_omitted_cell_index_resolves_to_last_cell(self):
        """An action that never spells cell_index falls back to the model default and hits the last cell."""
        action = FTTestAction(at_rollout=0, action="stop_cell_at_end")
        assert action.resolve_cell_id(_CELL_IDS) == "trainer-actor-2"


class TestRunAfterStep:
    @pytest.mark.asyncio
    async def test_stop_cell_fires_on_matching_rollout(self):
        """stop_cell_at_end triggers the worker manager with the resolved cell id on its rollout."""
        controller = FakeController(num_cells=3)
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller)

        with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
            await executor.run_after_step(5)

        assert manager.stopped == ["trainer-actor-1"]
        assert manager.started == []

    @pytest.mark.asyncio
    async def test_no_action_on_non_matching_rollout(self):
        """run_after_step does nothing when no action's at_rollout matches the given rollout."""
        controller = FakeController(num_cells=3)
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_index=1)
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller)

        with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
            await executor.run_after_step(4)

        assert manager.stopped == []
        assert manager.started == []

    @pytest.mark.asyncio
    async def test_start_cell_with_default_index_resolves_to_last_cell(self):
        """start_cell_at_end with cell_index -1 calls the worker manager on the last cell."""
        controller = FakeController(num_cells=3)
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=2, action="start_cell_at_end", cell_index=-1)
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller)

        with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
            await executor.run_after_step(2)

        assert manager.started == ["trainer-actor-2"]
        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_two_actions_same_rollout_both_fire(self):
        """Two actions sharing the same rollout both dispatch to their respective controller methods."""
        controller = FakeController(num_cells=3)
        manager = FakeWorkerManager()
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_index=0)
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_index=2)
        executor = FTTestActionControllerExecutor(actions=[stop_action, start_action], controller=controller)

        with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
            await executor.run_after_step(7)

        assert manager.stopped == ["trainer-actor-0"]
        assert manager.started == ["trainer-actor-2"]

    @pytest.mark.asyncio
    async def test_empty_actions_is_noop(self):
        """An executor with no actions performs no controller calls."""
        controller = FakeController(num_cells=3)
        manager = FakeWorkerManager()
        executor = FTTestActionControllerExecutor(actions=[], controller=controller)

        with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
            await executor.run_after_step(5)

        assert manager.stopped == []
        assert manager.started == []


_TWO_CELL_IDS = _CELL_IDS[:2]

_CRASH_ACTION = FTTestAction(at_rollout=4, action="crash_before_allreduce", cell_index=1, rank=0, attempt=0)


def _make_actor_executor(*, cell_id: str, rank: int) -> FTTestActionActorExecutor:
    return FTTestActionActorExecutor(actions=[_CRASH_ACTION], cell_id=cell_id, cell_ids=_TWO_CELL_IDS, rank=rank)


@pytest.fixture
def recorded_exit_codes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    exit_codes: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))
    return exit_codes


class TestMaybeCrash:
    def test_targeted_cell_and_rank_exits(self, recorded_exit_codes: list[int]) -> None:
        """The rank named by the action reaches os._exit(1) on the target rollout and attempt."""
        executor = _make_actor_executor(cell_id="trainer-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == [1]

    def test_other_cell_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        """A worker in a cell the action does not name keeps running."""
        executor = _make_actor_executor(cell_id="trainer-actor-0", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []

    def test_other_rank_in_targeted_cell_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        """Only the named rank of the named cell crashes, not its siblings."""
        executor = _make_actor_executor(cell_id="trainer-actor-1", rank=1)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []

    def test_other_rollout_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        """The crash is armed for one rollout only."""
        executor = _make_actor_executor(cell_id="trainer-actor-1", rank=0)

        executor.maybe_crash(rollout_id=3, attempt=0)

        assert recorded_exit_codes == []

    def test_other_attempt_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        """The retry after the injected crash must not crash again."""
        executor = _make_actor_executor(cell_id="trainer-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=1)

        assert recorded_exit_codes == []

    def test_default_cell_index_targets_the_last_cell(self, recorded_exit_codes: list[int]) -> None:
        """With the default cell_index the last cell crashes and the first one survives."""
        action = FTTestAction(at_rollout=4, action="crash_before_allreduce")
        last = FTTestActionActorExecutor(actions=[action], cell_id="trainer-actor-1", cell_ids=_TWO_CELL_IDS, rank=0)
        first = FTTestActionActorExecutor(actions=[action], cell_id="trainer-actor-0", cell_ids=_TWO_CELL_IDS, rank=0)

        first.maybe_crash(rollout_id=4, attempt=0)
        assert recorded_exit_codes == []

        last.maybe_crash(rollout_id=4, attempt=0)
        assert recorded_exit_codes == [1]
