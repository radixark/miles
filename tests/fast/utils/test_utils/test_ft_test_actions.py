import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.ft_test_actions import (
    _ACTOR_ACTIONS,
    _CONTROLLER_ACTIONS,
    FTTestAction,
    FTTestActionActorExecutor,
    FTTestActionControllerExecutor,
    _load_actions,
)

_POOL_ID = "trainer-actor"


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
    raw = json.dumps([{"at_rollout": 3, "action": "crash_before_allreduce", "cell_id": "trainer-actor-2"}])
    actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, FTTestAction)
    assert action.at_rollout == 3
    assert action.action == "crash_before_allreduce"
    assert action.cell_id == "trainer-actor-2"
    assert action.rank == 0
    assert action.attempt == 0


def test_load_actions_filters_to_only_matching_actions() -> None:
    """Mixed actions are filtered down to those whose action is in the filter set."""
    raw = json.dumps(
        [
            {"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-actor-0"},
            {"at_rollout": 2, "action": "crash_before_allreduce", "cell_id": "trainer-actor-1"},
            {"at_rollout": 3, "action": "start_cell_at_end", "cell_id": "trainer-actor-0"},
        ]
    )
    group_actions = _load_actions(_args(raw), _CONTROLLER_ACTIONS)
    assert [a.action for a in group_actions] == ["stop_cell_at_end", "start_cell_at_end"]
    actor_actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert [a.action for a in actor_actions] == ["crash_before_allreduce"]


def test_load_actions_returns_empty_when_no_action_matches_filter() -> None:
    """Valid actions that fall outside the filter set produce an empty result."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce", "cell_id": "trainer-actor-1"}])
    assert _load_actions(_args(raw), _CONTROLLER_ACTIONS) == []


def test_load_actions_rejects_extra_field() -> None:
    """An unexpected JSON field is rejected because the model forbids extras."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-actor-0", "bogus": 5}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_invalid_action_literal() -> None:
    """An action string outside the allowed Literal set raises a validation error."""
    raw = json.dumps([{"at_rollout": 1, "action": "not_a_real_action", "cell_id": "trainer-actor-0"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_missing_cell_id() -> None:
    """cell_id is required, so an action that omits it fails to load instead of guessing a target."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_legacy_cell_index_field() -> None:
    """The retired cell_index field is an extra field now, so stale JSON fails loudly."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_index": -1}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_cell_id_without_index_suffix() -> None:
    """A cell_id that carries no trailing index cannot be parsed and is rejected at load time."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "traineractor"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_cell_id_with_non_numeric_index() -> None:
    """A cell_id whose suffix is not an integer is rejected at load time."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-actor-last"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_validates_cell_id_of_actions_outside_the_filter() -> None:
    """Validation runs over every action, so a typo in another executor's action still fails here."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce", "cell_id": "bogus"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


class FakeController:
    def __init__(
        self,
        num_cells: int,
        *,
        pool_id: str = _POOL_ID,
        observed_after_reads: int = 0,
        gone_cell_id: str | None = None,
        gone_after_reads: int = 0,
    ) -> None:
        self.pool_id = pool_id
        self.expected_num_cells = num_cells
        self.cell_ids_reads = 0
        self._observed_after_reads = observed_after_reads
        self._gone_cell_id = gone_cell_id
        self._gone_after_reads = gone_after_reads

    @property
    def cell_ids(self) -> list[str]:
        self.cell_ids_reads += 1
        if self.cell_ids_reads <= self._observed_after_reads:
            return []
        cell_ids = [f"{self.pool_id}-{index}" for index in range(self.expected_num_cells)]
        if self._gone_cell_id is not None and self.cell_ids_reads > self._gone_after_reads:
            cell_ids = [cell_id for cell_id in cell_ids if cell_id != self._gone_cell_id]
        return cell_ids


class FakeRemoteMethod:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink
        self.error: Exception | None = None

    async def remote(self, cell_ids: list[str]) -> None:
        if self.error is not None:
            raise self.error
        self._sink.extend(cell_ids)


class FakeWorkerManager:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.started: list[str] = []
        self.stop_cells = FakeRemoteMethod(self.stopped)
        self.start_cells = FakeRemoteMethod(self.started)


async def _run(executor: FTTestActionControllerExecutor, manager: FakeWorkerManager, rollout_id: int) -> None:
    with patch("miles.utils.test_utils.ft_test_actions.RayWorkerManager.get_handle", lambda: manager):
        await executor.run_after_step(rollout_id)


class TestRunAfterStep:
    @pytest.mark.asyncio
    async def test_stop_cell_fires_on_matching_rollout(self):
        """stop_cell_at_end hands the action's cell_id to the worker manager on its rollout."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_id="trainer-actor-1")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3, gone_cell_id="trainer-actor-1")
        )

        await _run(executor, manager, 5)

        assert manager.stopped == ["trainer-actor-1"]
        assert manager.started == []

    @pytest.mark.asyncio
    async def test_no_action_on_non_matching_rollout(self):
        """run_after_step does nothing when no action's at_rollout matches the given rollout."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_id="trainer-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=3))

        await _run(executor, manager, 4)

        assert manager.stopped == []
        assert manager.started == []

    @pytest.mark.asyncio
    async def test_start_cell_targets_the_named_cell(self):
        """start_cell_at_end calls the worker manager with exactly the cell_id the action names."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=2, action="start_cell_at_end", cell_id="trainer-actor-2")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=3))

        await _run(executor, manager, 2)

        assert manager.started == ["trainer-actor-2"]
        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_start_cell_does_not_return_until_the_controller_observes_the_cell(self):
        """The next step reconfigures against what is observed, so returning early races the heal."""
        manager = FakeWorkerManager()
        controller = FakeController(num_cells=2, observed_after_reads=1)
        action = FTTestAction(at_rollout=3, action="start_cell_at_end", cell_id="trainer-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller)

        await _run(executor, manager, 3)

        assert manager.started == ["trainer-actor-1"]
        assert controller.cell_ids_reads > 1, "the resume returned on the read that still lacked the cell"

    @pytest.mark.asyncio
    async def test_stop_cell_does_not_return_while_the_controller_still_observes_the_cell(self):
        """A resume issued against a view that has not seen the suspend yet passes its own wait on that stale view."""
        manager = FakeWorkerManager()
        controller = FakeController(num_cells=2, gone_cell_id="trainer-actor-1", gone_after_reads=1)
        action = FTTestAction(at_rollout=3, action="stop_cell_at_end", cell_id="trainer-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller)

        await _run(executor, manager, 3)

        assert manager.stopped == ["trainer-actor-1"]
        assert controller.cell_ids_reads > 1, "the suspend returned on the read that still showed the cell"

    @pytest.mark.asyncio
    async def test_start_cell_after_that_cell_was_dropped_still_targets_it(self):
        """A stopped cell no longer being live does not change the cell_id the action names."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=3, action="start_cell_at_end", cell_id="trainer-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=2))

        await _run(executor, manager, 3)

        assert manager.started == ["trainer-actor-1"]
        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_two_actions_same_rollout_both_fire(self):
        """Two actions sharing the same rollout both dispatch to their respective controller methods."""
        manager = FakeWorkerManager()
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_id="trainer-actor-0")
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_id="trainer-actor-2")
        executor = FTTestActionControllerExecutor(
            actions=[stop_action, start_action],
            controller=FakeController(num_cells=3, gone_cell_id="trainer-actor-0"),
        )

        await _run(executor, manager, 7)

        assert manager.stopped == ["trainer-actor-0"]
        assert manager.started == ["trainer-actor-2"]

    @pytest.mark.asyncio
    async def test_empty_actions_is_noop(self):
        """An executor with no actions performs no controller calls."""
        manager = FakeWorkerManager()
        executor = FTTestActionControllerExecutor(actions=[], controller=FakeController(num_cells=3))

        await _run(executor, manager, 5)

        assert manager.stopped == []
        assert manager.started == []

    @pytest.mark.asyncio
    async def test_action_naming_another_spec_raises(self):
        """An action aimed at a different spec is a misconfiguration and must fail, not silently no-op."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="rollout-engine-0")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=3))

        with pytest.raises(AssertionError):
            await _run(executor, manager, 1)

        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_action_index_beyond_expected_num_cells_raises(self):
        """A cell index the group can never have is a misconfiguration and must fail at dispatch."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="trainer-actor-9")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=3))

        with pytest.raises(AssertionError):
            await _run(executor, manager, 1)

        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_the_index_one_past_the_last_cell_is_rejected(self):
        """Cell indices are half-open, so index N of an N-cell pool is the easiest off-by-one to write in CI config."""
        manager = FakeWorkerManager()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="trainer-actor-3")
        executor = FTTestActionControllerExecutor(actions=[action], controller=FakeController(num_cells=3))

        with pytest.raises(AssertionError):
            await _run(executor, manager, 1)

        assert manager.stopped == []

    @pytest.mark.asyncio
    async def test_a_rejected_stop_propagates_and_the_later_action_never_fires(self):
        """Carrying on after the requested transition failed turns a broken scenario into a green run."""
        manager = FakeWorkerManager()
        manager.stop_cells.error = RuntimeError("worker manager rejected the stop")
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_id="trainer-actor-0")
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_id="trainer-actor-2")
        executor = FTTestActionControllerExecutor(
            actions=[stop_action, start_action], controller=FakeController(num_cells=3)
        )

        with pytest.raises(RuntimeError, match="rejected the stop"):
            await _run(executor, manager, 7)

        assert manager.stopped == []
        assert manager.started == []


_CRASH_ACTION = FTTestAction(
    at_rollout=4, action="crash_before_allreduce", cell_id="trainer-actor-1", rank=0, attempt=0
)


def _make_actor_executor(*, cell_id: str, rank: int) -> FTTestActionActorExecutor:
    return FTTestActionActorExecutor(actions=[_CRASH_ACTION], cell_id=cell_id, rank=rank)


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

    def test_cell_of_another_spec_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        """Matching is exact string equality, so a same-index cell of another spec survives."""
        executor = _make_actor_executor(cell_id="rollout-engine-1", rank=0)

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

    def test_no_actions_never_exits(self, recorded_exit_codes: list[int]) -> None:
        """An actor executor with no actions never crashes its worker."""
        executor = FTTestActionActorExecutor(actions=[], cell_id="trainer-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []
