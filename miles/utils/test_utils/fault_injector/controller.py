import asyncio
import threading
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.models import (
    FaultHookName,
    FaultHookOwner,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
)
from miles.utils.test_utils.fault_injector.request_executor import FaultHookRequestExecutor
from miles.utils.test_utils.fault_injector.static_source import read_declared_fault_hooks


class FaultHookConflictError(Exception):
    pass


class FaultHookOperation(StrEnum):
    SET = "set"
    CLEAR = "clear"


class FaultHookCommand(FrozenStrictBaseModel):
    operation: FaultHookOperation
    request: FaultHookRequest


def reach_fault_hook(hook_name: FaultHookName, **context: int | str | None) -> None:
    fault_hook_controller._reach(hook_name, context)


async def reach_fault_hook_async(hook_name: FaultHookName, **context: int | str | None) -> None:
    await fault_hook_controller._reach_async(hook_name, context)


class _FaultHookController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executors: dict[str, FaultHookRequestExecutor] = {}
        self._context: FaultHookContext | None = None
        self._resources = FaultHookResources()

    def configure(
        self,
        *,
        resources: FaultHookResources,
        owner: FaultHookOwner | None = None,
        cell_id: str | None = None,
        rank: int | None = None,
    ) -> None:
        self._resources = resources
        if owner is None:
            return
        assert resources.args is not None, "A process that reaches fault hooks reads its declared hooks from args"
        declared = read_declared_fault_hooks(resources.args)
        for request in _filter_fault_hooks(declared, owner=owner, cell_id=cell_id, rank=rank):
            self.apply(FaultHookCommand(operation=FaultHookOperation.SET, request=request))

    @contextmanager
    def with_context(self, context: FaultHookContext) -> Iterator[None]:
        self._context = context
        try:
            yield
        finally:
            self._context = None

    def apply(self, command: FaultHookCommand) -> FaultHookRecord:
        with self._lock:
            self._drop_expired()
            match command.operation:
                case FaultHookOperation.SET:
                    executor = self._set(command.request)
                case FaultHookOperation.CLEAR:
                    return self._clear(command.request)
            fired = (
                self._dispatch(executor, context=self._current_context({}))
                if command.request.hook_name is None
                else None
            )
        if fired is not None:
            _run_blocking(fired.execute(resources=self._resources))
        return executor.record

    def _set(self, request: FaultHookRequest) -> FaultHookRequestExecutor:
        if request.request_id in self._executors or any(
            executor.record.request.conflicts_with(request) for executor in self._executors.values()
        ):
            raise FaultHookConflictError(f"A fault hook is already set for the same trigger: {request.request_id}")
        self._executors[request.request_id] = executor = FaultHookRequestExecutor(request)
        return executor

    def _clear(self, request: FaultHookRequest) -> FaultHookRecord:
        if (executor := self._executors.get(request.request_id)) is None:
            raise FaultHookConflictError("Fault hook clearing names a request this process never set")
        if executor.record.request != request:
            raise FaultHookConflictError("Fault hook clearing does not match the original request")
        del self._executors[request.request_id]
        return executor.clear()

    def _reach(self, hook_name: FaultHookName, context: dict[str, int | str | None]) -> None:
        if fired := self._dispatch_reached(hook_name, context):
            _run_blocking(self._execute(fired))

    async def _reach_async(self, hook_name: FaultHookName, context: dict[str, int | str | None]) -> None:
        await self._execute(self._dispatch_reached(hook_name, context))

    async def _execute(self, executors: list[FaultHookRequestExecutor]) -> None:
        for executor in executors:
            await executor.execute(resources=self._resources)

    def _dispatch_reached(
        self, hook_name: FaultHookName, context: dict[str, int | str | None]
    ) -> list[FaultHookRequestExecutor]:
        reached_context = self._current_context(context)
        with self._lock:
            self._drop_expired()
            candidates = [
                executor
                for executor in self._executors.values()
                if executor.record.status == FaultHookStatus.PENDING
                and executor.record.request.hook_name == hook_name
                and executor.record.request.matches(reached_context)
            ]
            return [fired for executor in candidates if (fired := self._dispatch(executor, context=reached_context))]

    def _dispatch(
        self, executor: FaultHookRequestExecutor, *, context: FaultHookContext
    ) -> FaultHookRequestExecutor | None:
        executor.mark_reached(context=context)
        if executor.record.request.delay_ms > 0:
            executor.schedule(on_due=self._on_due)
            return None
        del self._executors[executor.record.request.request_id]
        return executor.mark_fired()

    def _current_context(self, context: dict[str, int | str | None]) -> FaultHookContext:
        return (self._context or FaultHookContext()).model_copy(update=context)

    def _on_due(self, executor: FaultHookRequestExecutor) -> None:
        with self._lock:
            self._drop_expired()
            if self._executors.get(executor.record.request.request_id) is not executor:
                return
            del self._executors[executor.record.request.request_id]
            executor.mark_fired()

        _run_blocking(executor.execute(resources=self._resources))

    def _drop_expired(self) -> None:
        for request_id, executor in list(self._executors.items()):
            if executor.is_expired():
                del self._executors[request_id]
                executor.expire()


def _filter_fault_hooks(
    requests: list[FaultHookRequest], *, owner: FaultHookOwner, cell_id: str | None, rank: int | None
) -> list[FaultHookRequest]:
    return [
        request
        for request in requests
        if (request.hook_name is None or request.hook_name.owner == owner)
        and request.target.covers(cell_id=cell_id, rank=rank)
    ]


def _run_blocking(coroutine: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
        return
    runner = threading.Thread(target=asyncio.run, args=(coroutine,), daemon=True)
    runner.start()
    runner.join()


fault_hook_controller = _FaultHookController()
