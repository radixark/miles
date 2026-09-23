import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.test_utils.fault_injector.actions.frozen import (
    SLEEP_FOREVER_AT_END_ACTION,
    assert_loop_parkable,
    write_frozen_sentinel,
)
from miles.utils.test_utils.fault_injector.static_source import read_declared_actions
from miles.utils.workers.naming import parse_cell_id

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations

logger = logging.getLogger(__name__)

SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0

_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS = 300.0

_CONTROLLER_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ORCHESTRATION_ACTIONS = {SLEEP_FOREVER_AT_END_ACTION}

SleepFn = Callable[[float], Awaitable[None]]


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "sleep_forever_at_end"]
    cell_id: str | None = None

    @model_validator(mode="after")
    def _check_cell_name_matches_action(self) -> "FTTestAction":
        assert (self.action in _ORCHESTRATION_ACTIONS) == (self.cell_id is None), (
            f"an orchestration action names no cell and a cell action names one, and {self.action} names "
            f"cell_id={self.cell_id!r}"
        )
        return self


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    if not (raw := read_declared_actions(args)):
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
            assert_loop_parkable(args, trainer_model_id=trainer_model_id)

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
