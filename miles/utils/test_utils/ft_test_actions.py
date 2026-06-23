import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce"]
    cell_index: int = -1  # -1 = last cell
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)

    def resolve_cell_index(self, num_cells: int) -> int:
        return self.cell_index if self.cell_index >= 0 else num_cells - 1


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])

_GROUP_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
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


class FTTestActionGroupExecutor:
    def __init__(self, *, actions: list[FTTestAction], group: "RayTrainGroup") -> None:
        self._actions = actions
        self._group = group

    @staticmethod
    def from_args(args: object, *, group: "RayTrainGroup") -> "FTTestActionGroupExecutor":
        return FTTestActionGroupExecutor(actions=_load_actions(args, _GROUP_ACTIONS), group=group)

    def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.at_rollout == rollout_id:
                cell_index = action.resolve_cell_index(self._group.num_cells)
                logger.info("FT test action: %s cell %d after rollout %d", action.action, cell_index, rollout_id)

                if action.action == "stop_cell_at_end":
                    self._group.stop_cell(cell_index)
                elif action.action == "start_cell_at_end":
                    self._group.start_cell(cell_index)


class FTTestActionActorExecutor:
    def __init__(self, *, actions: list[FTTestAction], cell_index: int, num_cells: int, rank: int) -> None:
        self._actions = actions
        self._cell_index = cell_index
        self._num_cells = num_cells
        self._rank = rank

    @staticmethod
    def from_args(
        args: object,
        *,
        cell_index: int,
        num_cells: int,
        rank: int,
    ) -> "FTTestActionActorExecutor":
        return FTTestActionActorExecutor(
            actions=_load_actions(args, _ACTOR_ACTIONS),
            cell_index=cell_index,
            num_cells=num_cells,
            rank=rank,
        )

    def maybe_crash(self, *, rollout_id: int, attempt: int) -> None:
        for action in self._actions:
            if (
                action.at_rollout == rollout_id
                and action.attempt == attempt
                and action.resolve_cell_index(self._num_cells) == self._cell_index
                and action.rank == self._rank
            ):
                msg = (
                    f"FT test action: crash_before_allreduce at rollout {rollout_id} "
                    f"attempt {attempt} cell {self._cell_index} rank {self._rank} — calling os._exit(1)"
                )
                logger.warning(msg)
                print(msg, flush=True)
                os._exit(1)
