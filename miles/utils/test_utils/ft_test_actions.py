import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.frozen import (
    SLEEP_FOREVER_AT_END_ACTION,
    assert_loop_parkable,
    write_frozen_sentinel,
)
from miles.utils.test_utils.fault_injector.static_source import read_declared_actions

logger = logging.getLogger(__name__)

SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0

_ORCHESTRATION_ACTIONS = {SLEEP_FOREVER_AT_END_ACTION}

SleepFn = Callable[[float], Awaitable[None]]


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["sleep_forever_at_end"]


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    if not (raw := read_declared_actions(args)):
        return []
    all_actions = _ACTION_LIST_ADAPTER.validate_json(raw)

    actions = [a for a in all_actions if a.action in action_filter]
    if actions:
        logger.info("FT test actions activated: %d actions (%s)", len(actions), action_filter)
    return actions


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
