import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.args.runtime_base import BaseLeafConfig


@dataclass
class RolloutRuntimeState:
    sglang_router_ip: str | None = None
    sglang_router_port: int | None = None
    sglang_model_routers: dict[str, tuple[str, int]] = field(default_factory=dict)
    session_server_addrs: list[str] = field(default_factory=list)
    session_server_instance_ids: dict[str, str] = field(default_factory=dict)
    rollout_engine_count: int = 0
    rollout_gpu_count: int = 0
    eval_engine_count: int = 0


def compute_rollout_runtime_config(
    args: BaseLeafConfig | ImmutableNamespace, runtime: RolloutRuntimeState
) -> ImmutableNamespace:
    return ImmutableNamespace.model_validate(dict(args) | vars(runtime) | {"runtime": runtime})


class GenerationConcurrencyLimiter:
    def __init__(self, capacity: Callable[[], int]) -> None:
        self._capacity = capacity
        self._condition = asyncio.Condition()
        self._active = 0

    async def __aenter__(self) -> Self:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < max(self._capacity(), 1))
            self._active += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()
