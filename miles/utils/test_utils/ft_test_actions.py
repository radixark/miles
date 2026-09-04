import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.ray_worker_manager import RayWorkerManager

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce"]
    cell_id: str
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])

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
    def __init__(self, *, actions: list[FTTestAction], controller: "TrainerController") -> None:
        self._actions = actions
        self._controller = controller

    @staticmethod
    def from_args(args: object, *, controller: "TrainerController") -> "FTTestActionControllerExecutor":
        return FTTestActionControllerExecutor(actions=_load_actions(args, _CONTROLLER_ACTIONS), controller=controller)

    async def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.at_rollout == rollout_id:
                self._check_action_target(action)
                logger.info("FT test action: %s cell %s after rollout %d", action.action, action.cell_id, rollout_id)

                worker_manager = RayWorkerManager.get_handle()
                if action.action == "stop_cell_at_end":
                    await worker_manager.stop_cells.remote([action.cell_id])
                elif action.action == "start_cell_at_end":
                    await worker_manager.start_cells.remote([action.cell_id])

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
