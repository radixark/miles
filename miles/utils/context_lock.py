import asyncio
import contextvars
import functools
import inspect
from collections.abc import Callable
from types import TracebackType
from typing import Any

LOCK_ATTRIBUTE_NAME: str = "context_lock"

_held_lock: contextvars.ContextVar["ContextLock | None"] = contextvars.ContextVar("held_context_lock", default=None)


class ContextLock:
    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def held_in_current_context(self) -> bool:
        return _held_lock.get() is self

    async def __aenter__(self) -> "ContextLock":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    async def acquire(self) -> None:
        assert _held_lock.get() is None, f"Cannot acquire lock {self._name!r}: a context lock is already held"
        await self._lock.acquire()
        _held_lock.set(self)

    def release(self) -> None:
        assert _held_lock.get() is self, f"Lock {self._name!r} must be held by the current context"
        _held_lock.set(None)
        self._lock.release()


def with_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use with_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async with _get_lock(self):
            return await fn(self, *args, **kwargs)

    return wrapper


def _get_lock(obj: Any) -> ContextLock:
    lock = getattr(obj, LOCK_ATTRIBUTE_NAME)
    assert isinstance(lock, ContextLock), f"{type(obj).__name__}.{LOCK_ATTRIBUTE_NAME} must be a ContextLock"
    return lock
