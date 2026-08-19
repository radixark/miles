import json
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.naming import parse_cell_id

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce"]
    cell_id: str
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)


CI_FT_TEST_ACTIONS_FLAG: str = "--ci-ft-test-actions"

_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def compute_ft_test_actions_arg(actions: Sequence[dict]) -> str:
    return f"{CI_FT_TEST_ACTIONS_FLAG} '{render_ft_test_actions(actions)}' "


def render_ft_test_actions(actions: Sequence[dict]) -> str:
    return json.dumps(list(actions))


_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS = 300.0

_CONTROLLER_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ACTOR_ACTIONS = {"crash_before_allreduce"}


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    raw: str | None = getattr(args, "ci_ft_test_actions", None)
    if not raw:
        return []
    all_actions = _ACTION_LIST_ADAPTER.validate_json(raw)

    for action in all_actions:
        try:
            parse_cell_id(action.cell_id)
        except ValueError as e:
            raise ValueError(f"FT test action has malformed cell_id {action.cell_id!r} (action={action})") from e

    actions = [a for a in all_actions if a.action in action_filter]
    if actions:
        logger.info("FT test actions activated: %d actions (%s)", len(actions), action_filter)
    return actions


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
                elif action.action == "start_cell_at_end":
                    await operations.resume(cell_id=action.cell_id)
                    await self._wait_cell_observed(action.cell_id)

    async def _wait_cell_observed(self, cell_id: str) -> None:
        async def _check(_remaining: float) -> None:
            if cell_id not in self._controller.cell_ids:
                raise TimeoutError(f"{cell_id} was resumed but is not observed yet")

        await retry_until_deadline(
            _check,
            total_seconds=_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", op="wait_cell_observed", cell=cell_id),
        )

    def _check_action_target(self, action: FTTestAction) -> None:
        parsed = parse_cell_id(action.cell_id)
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
