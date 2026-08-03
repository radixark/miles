import asyncio
import contextvars
import functools
import inspect
import logging
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

WAIT_LOG_INTERVAL_SECONDS: float = 5.0
LOCK_ATTRIBUTE_NAME: str = "context_lock"

_DISCIPLINE_MARKER_ATTRIBUTE_NAME: str = "_context_lock_discipline"

# the annotation machinery (PEP 649) plants these in the class dict; they are not methods of the class
_ANNOTATION_MEMBER_NAMES: frozenset[str] = frozenset({"__annotate__", "__annotate_func__"})

_held_lock: contextvars.ContextVar["ContextLock | None"] = contextvars.ContextVar("held_context_lock", default=None)


class ContextLock:
    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = asyncio.Lock()
        self._detached = False

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
        wait_reminder = asyncio.ensure_future(self._remind_while_waiting())
        try:
            await self._lock.acquire()
        finally:
            wait_reminder.cancel()
        _held_lock.set(self)

    def release(self) -> None:
        self._assert_held_in_current_context()
        _held_lock.set(None)
        self._lock.release()

    def detach(self) -> None:
        self._assert_held_in_current_context()
        self._detached = True
        _held_lock.set(None)

    def reattach(self) -> None:
        assert self._detached, f"Cannot reattach lock {self._name!r}: it was not detached"
        assert _held_lock.get() is None, f"Cannot reattach lock {self._name!r}: a context lock is already held"
        self._detached = False
        _held_lock.set(self)

    def _assert_held_in_current_context(self) -> None:
        assert _held_lock.get() is self, f"Lock {self._name!r} must be held by the current context"

    async def _remind_while_waiting(self) -> None:
        wait_start_time = time.monotonic()
        while True:
            await asyncio.sleep(WAIT_LOG_INTERVAL_SECONDS)
            logger.info(f"Still waiting for lock {self._name!r} after {time.monotonic() - wait_start_time:.0f}s")


def enforce_lock_discipline(cls: type) -> type:
    for member_name, member in vars(cls).items():
        if member_name in _ANNOTATION_MEMBER_NAMES:
            continue
        for fn in _extract_checkable_functions(member):
            assert getattr(fn, _DISCIPLINE_MARKER_ATTRIBUTE_NAME, None) is not None, (
                f"{cls.__name__}.{member_name} must be decorated with one of the context-lock decorators "
                f"(e.g. with_lock or lock_exempt)"
            )
    return cls


def with_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use with_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async with _get_lock(self):
            return await fn(self, *args, **kwargs)

    return _mark(wrapper, "with_lock")


def acquires_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use acquires_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = _get_lock(self)
        await lock.acquire()
        try:
            result = await fn(self, *args, **kwargs)
        except BaseException:
            lock.release()
            raise
        lock.detach()
        return result

    return _mark(wrapper, "acquires_lock")


def releases_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use releases_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = _get_lock(self)
        lock.reattach()
        try:
            return await fn(self, *args, **kwargs)
        finally:
            lock.release()

    return _mark(wrapper, "releases_lock")


def requires_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    def assert_precondition(self: Any) -> None:
        _assert_own_lock_held(fn, self)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            assert_precondition(self)
            return await fn(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            assert_precondition(self)
            return fn(self, *args, **kwargs)

    return _mark(wrapper, "requires_lock")


def lock_exempt(fn: Callable[..., Any]) -> Callable[..., Any]:
    return _mark(fn, "lock_exempt")


def _extract_checkable_functions(member: Any) -> list[Callable[..., Any]]:
    if isinstance(member, (staticmethod, classmethod)):
        return [member.__func__]
    if isinstance(member, property):
        return [accessor for accessor in (member.fget, member.fset, member.fdel) if accessor is not None]
    if inspect.isfunction(member):
        return [member]
    return []


def _get_lock(obj: Any) -> ContextLock:
    lock = getattr(obj, LOCK_ATTRIBUTE_NAME)
    assert isinstance(lock, ContextLock), f"{type(obj).__name__}.{LOCK_ATTRIBUTE_NAME} must be a ContextLock"
    return lock


def _mark(fn: Callable[..., Any], discipline: str) -> Callable[..., Any]:
    setattr(fn, _DISCIPLINE_MARKER_ATTRIBUTE_NAME, discipline)
    return fn


def _assert_own_lock_held(fn: Callable[..., Any], obj: Any) -> None:
    lock = _get_lock(obj)
    assert (
        lock.held_in_current_context
    ), f"{fn.__qualname__} must be called with the {lock.name!r} context lock held by the current context"
