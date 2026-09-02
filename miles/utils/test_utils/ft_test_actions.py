import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.naming import parse_cell_id

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations

logger = logging.getLogger(__name__)

CI_FT_TEST_ACTIONS_FLAG: str = "--ci-ft-test-actions"
SLEEP_FOREVER_AT_END_ACTION: str = "sleep_forever_at_end"
SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0
PARKABLE_TRAIN_SCRIPT: str = "train.py"


def compute_ft_test_actions_arg(actions: Sequence[dict]) -> str:
    return f"{CI_FT_TEST_ACTIONS_FLAG} '{render_ft_test_actions(actions)}' "


def render_ft_test_actions(actions: Sequence[dict]) -> str:
    return json.dumps(list(actions))


_CELL_OBSERVED_TIMEOUT_SECONDS = 300.0

_CONTROLLER_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ACTOR_ACTIONS = {"crash_before_allreduce"}
_ORCHESTRATION_ACTIONS = {SLEEP_FOREVER_AT_END_ACTION}

SleepFn = Callable[[float], Awaitable[None]]


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce", "sleep_forever_at_end"]
    cell_id: str | None = None
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)

    @model_validator(mode="after")
    def _check_a_cell_is_named_exactly_when_the_action_acts_on_one(self) -> "FTTestAction":
        assert (self.action in _ORCHESTRATION_ACTIONS) == (self.cell_id is None), (
            f"an orchestration action names no cell and a cell action names one, and {self.action} names "
            f"cell_id={self.cell_id!r}"
        )
        return self


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    if not (raw := _read_declared_actions(args)):
        return []
    all_actions = _ACTION_LIST_ADAPTER.validate_json(raw)

    for action in all_actions:
        if (cell_id := action.cell_id) is None:
            continue
        try:
            parse_cell_id(cell_id)
        except ValueError as e:
            raise ValueError(f"FT test action has malformed cell_id {cell_id!r} (action={action})") from e

    actions = [a for a in all_actions if a.action in action_filter]
    if actions:
        logger.info("FT test actions activated: %d actions (%s)", len(actions), action_filter)
    return actions


def _assert_loop_parkable(args: object, *, trainer_model_id: str | None) -> None:
    assert (script := Path(sys.argv[0]).name) == PARKABLE_TRAIN_SCRIPT, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the orchestration script between two steps, and only "
        f"{PARKABLE_TRAIN_SCRIPT} stands still at that point; {script} has already started the next rollout by the "
        f"time it reaches here, so the run would not be standing where the action names"
    )
    assert (interval := args.update_weights_interval) == 1, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the run where it updates weights, and --update-weights-interval "
        f"{interval} means the run does not pass through that point after every step"
    )
    assert trainer_model_id is None, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks one coroutine, and a run training several policies drives one per "
        f"policy ({trainer_model_id!r} reached it here), so every other policy would keep training past the step "
        f"the action names"
    )


class FTTestActionControllerExecutor:
    def __init__(
        self, *, actions: list[FTTestAction], controller: "TrainerController", cell_operations: "BaseCellOperations"
    ) -> None:
        self._actions = actions
        self._controller = controller
        self._cell_operations = cell_operations

    @staticmethod
    def from_args(
        args: object, *, controller: "TrainerController", cell_operations: "BaseCellOperations"
    ) -> "FTTestActionControllerExecutor":
        return FTTestActionControllerExecutor(
            actions=_load_actions(args, _CONTROLLER_ACTIONS), controller=controller, cell_operations=cell_operations
        )

    async def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.at_rollout == rollout_id:
                self._check_action_target(action)
                logger.info("FT test action: %s cell %s after rollout %d", action.action, action.cell_id, rollout_id)

                operations = self._cell_operations
                if action.action == "stop_cell_at_end":
                    await operations.suspend(cell_id=action.cell_id)
                    await self._wait_cell_observed(action.cell_id, observed=False)
                elif action.action == "start_cell_at_end":
                    await operations.resume(cell_id=action.cell_id)
                    await self._wait_cell_observed(action.cell_id, observed=True)

    async def _wait_cell_observed(self, cell_id: str, *, observed: bool) -> None:
        async def _check(_remaining: float) -> None:
            if (cell_id in self._controller.cell_ids) != observed:
                raise TimeoutError(f"{cell_id} is not observed as {'present' if observed else 'gone'} yet")

        await retry_until_deadline(
            _check,
            total_seconds=_CELL_OBSERVED_TIMEOUT_SECONDS,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", op="wait_cell_observed", cell=cell_id, observed=observed),
        )

    def _check_action_target(self, action: FTTestAction) -> None:
        assert (cell_id := action.cell_id) is not None
        parsed = parse_cell_id(cell_id)
        assert parsed.pool_id == self._controller.pool_id, (
            f"FT test action targets pool_id {parsed.pool_id!r} but this controller drives {self._controller.pool_id!r} "
            f"(action={action})"
        )
        assert parsed.cell_index < self._controller.expected_num_cells, (
            f"FT test action targets cell index {parsed.cell_index} but the pool only has "
            f"{self._controller.expected_num_cells} cells (action={action})"
        )


class FTTestActionActorExecutor:
    def __init__(self, *, actions: list[FTTestAction], cell_id: str, rank: int) -> None:
        self._actions = actions
        self._cell_id = cell_id
        self._rank = rank

    @staticmethod
    def from_args(
        args: object,
        *,
        cell_id: str,
        rank: int,
    ) -> "FTTestActionActorExecutor":
        return FTTestActionActorExecutor(
            actions=_load_actions(args, _ACTOR_ACTIONS),
            cell_id=cell_id,
            rank=rank,
        )

    def maybe_crash(self, *, rollout_id: int, attempt: int) -> None:
        for action in self._actions:
            if (
                action.at_rollout == rollout_id
                and action.attempt == attempt
                and action.cell_id == self._cell_id
                and action.rank == self._rank
            ):
                msg = (
                    f"FT test action: crash_before_allreduce at rollout {rollout_id} "
                    f"attempt {attempt} cell {self._cell_id} rank {self._rank} — calling os._exit(1)"
                )
                logger.warning(msg)
                print(msg, flush=True)
                os._exit(1)


class FTTestActionOrchestrationExecutor:
    def __init__(
        self,
        *,
        actions: list[FTTestAction],
        sleep: SleepFn = asyncio.sleep,
        interval_seconds: float = SLEEP_FOREVER_INTERVAL_SECONDS,
        actions_path: Path | None = None,
    ) -> None:
        self._actions = actions
        self._sleep = sleep
        self._interval_seconds = interval_seconds
        self._actions_path = actions_path

    @staticmethod
    def from_args(args: object, *, trainer_model_id: str | None = None) -> "FTTestActionOrchestrationExecutor":
        actions = _load_actions(args, _ORCHESTRATION_ACTIONS)
        if actions:
            _assert_loop_parkable(args, trainer_model_id=trainer_model_id)

        path: str | None = args.ci_ft_test_actions_path
        return FTTestActionOrchestrationExecutor(
            actions=actions,
            actions_path=Path(path) if path is not None else None,
        )

    async def run_after_step(self, rollout_id: int) -> None:
        actions = [action for action in self._actions if action.at_rollout == rollout_id]
        if not actions:
            return
        for action in actions:
            assert action.action == SLEEP_FOREVER_AT_END_ACTION, (
                f"the orchestration side runs {SLEEP_FOREVER_AT_END_ACTION} and nothing else, and {action.action} "
                f"reached it (action={action})"
            )

        msg = (
            f"FT test action: {SLEEP_FOREVER_AT_END_ACTION} at rollout {rollout_id} — this orchestration script "
            f"sleeps from here on and never starts rollout {rollout_id + 1}"
        )
        logger.warning(msg)
        print(msg, flush=True)
        if self._actions_path is not None:
            write_frozen_sentinel(self._actions_path, rollout_id=rollout_id)
        await self._sleep_forever()

    async def _sleep_forever(self) -> None:
        while True:
            await self._sleep(self._interval_seconds)


# ============ adhoc file delivery (revert after the args refactor) ============


CI_FT_TEST_ACTIONS_PATH_FLAG: str = "--ci-ft-test-actions-path"


# TODO ad hoc hack: revert after the args refactor
def _read_declared_actions(args: object) -> str:
    inline: str | None = args.ci_ft_test_actions
    path: str | None = args.ci_ft_test_actions_path

    assert inline is None or path is None, (
        f"{CI_FT_TEST_ACTIONS_FLAG} and {CI_FT_TEST_ACTIONS_PATH_FLAG} both name the actions a run performs, and a "
        f"run given both silently follows one of them"
    )
    return read_ft_test_actions(Path(path)) if path is not None else (inline or "")


# TODO ad hoc hack: revert after the args refactor
def write_ft_test_actions(path: Path, actions: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(render_ft_test_actions(actions))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_ft_test_actions(path: Path) -> str:
    stamp = _stat_or_none(path)
    assert stamp is not None, (
        f"{CI_FT_TEST_ACTIONS_PATH_FLAG} names {path}, which does not exist; a run told to read its plan from a "
        f"file nothing wrote would quietly perform no action at all"
    )

    stamped_at = (stamp.st_mtime_ns, stamp.st_size)
    if (cached := _ACTIONS_OF_STAMP.get(path)) is not None and cached[0] == stamped_at:
        return cached[1]

    text = path.read_text()
    _ACTIONS_OF_STAMP[path] = (stamped_at, text)
    return text


# TODO ad hoc hack: revert after the args refactor
def _stat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


FROZEN_SENTINEL_SUFFIX: str = "_frozen_at.json"
# TODO ad hoc hack: revert after the args refactor
_ACTIONS_OF_STAMP: dict[Path, tuple[tuple[int, int], str]] = {}


# TODO ad hoc hack: revert after the args refactor
def compute_frozen_sentinel_path(actions_path: Path) -> Path:
    return actions_path.with_name(f"{actions_path.stem}{FROZEN_SENTINEL_SUFFIX}")


# TODO ad hoc hack: revert after the args refactor
def write_frozen_sentinel(actions_path: Path, *, rollout_id: int) -> None:
    path = compute_frozen_sentinel_path(actions_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(json.dumps({"rollout_id": rollout_id}))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_frozen_rollout_id(actions_path: Path) -> int | None:
    path = compute_frozen_sentinel_path(actions_path)
    if not path.is_file():
        return None
    return int(json.loads(path.read_text())["rollout_id"])
