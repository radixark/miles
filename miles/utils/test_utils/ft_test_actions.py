import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.controller import RayTrainGroup

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce"]
    cell_index: int = -1  # -1 = last cell
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)

    def resolve_cell_id(self, cell_ids: list[str]) -> str:
        return cell_ids[self.cell_index]


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])

_CONTROLLER_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ACTOR_ACTIONS = {"crash_before_allreduce"}


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    raw: str | None = getattr(args, "ci_ft_test_actions", None)
    if not raw:
        return []
    all_actions = _ACTION_LIST_ADAPTER.validate_json(raw)
    actions = [a for a in all_actions if a.action in action_filter]
    if actions:
        logger.info("FT test actions activated: %d actions (%s)", len(actions), action_filter)
    return actions


class FTTestActionControllerExecutor:
    def __init__(self, *, actions: list[FTTestAction], controller: "RayTrainGroup") -> None:
        self._actions = actions
        self._controller = controller

    @staticmethod
    def from_args(args: object, *, controller: "RayTrainGroup") -> "FTTestActionControllerExecutor":
        return FTTestActionControllerExecutor(actions=_load_actions(args, _CONTROLLER_ACTIONS), controller=controller)

    async def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.at_rollout == rollout_id:
                cell_id = action.resolve_cell_id(self._controller.cell_ids)
                logger.info("FT test action: %s cell %s after rollout %d", action.action, cell_id, rollout_id)

                if action.action == "stop_cell_at_end":
                    await self._controller.stop_cell(cell_id)
                elif action.action == "start_cell_at_end":
                    self._controller.start_cell(cell_id)


class FTTestActionActorExecutor:
    def __init__(self, *, actions: list[FTTestAction], cell_id: str, cell_ids: list[str], rank: int) -> None:
        self._actions = actions
        self._cell_id = cell_id
        self._cell_ids = cell_ids
        self._rank = rank

    @staticmethod
    def from_args(
        args: object,
        *,
        cell_id: str,
        cell_ids: list[str],
        rank: int,
    ) -> "FTTestActionActorExecutor":
        return FTTestActionActorExecutor(
            actions=_load_actions(args, _ACTOR_ACTIONS),
            cell_id=cell_id,
            cell_ids=cell_ids,
            rank=rank,
        )

    def maybe_crash(self, *, rollout_id: int, attempt: int) -> None:
        for action in self._actions:
            if (
                action.at_rollout == rollout_id
                and action.attempt == attempt
                and action.resolve_cell_id(self._cell_ids) == self._cell_id
                and action.rank == self._rank
            ):
                msg = (
                    f"FT test action: crash_before_allreduce at rollout {rollout_id} "
                    f"attempt {attempt} cell {self._cell_id} rank {self._rank} — calling os._exit(1)"
                )
                logger.warning(msg)
                print(msg, flush=True)
                os._exit(1)
