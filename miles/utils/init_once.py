import contextlib
import enum
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

_T = TypeVar("_T", bound=Callable[..., Any])


class InitState(enum.Enum):
    NOT_INITED = "not_inited"
    INITIALIZING = "initializing"
    INITED = "inited"
    INIT_FAILED = "init_failed"


class InitOnce:
    def __init__(self, debug_owner_name: str) -> None:
        self._debug_owner_name = debug_owner_name
        self._state = InitState.NOT_INITED

    @property
    def state(self) -> InitState:
        return self._state

    def is_initialized(self) -> bool:
        return self._state is InitState.INITED

    @contextlib.contextmanager
    def guarding(self) -> Iterator[None]:
        self._enter()
        try:
            yield
        except BaseException:
            self._leave(succeeded=False)
            raise
        self._leave(succeeded=True)

    def _enter(self) -> None:
        assert self._state is InitState.NOT_INITED, (
            f"{self._debug_owner_name} already ran init in this process and is now {self._state.value}, so this is "
            f"a stale worker being reused as a fresh one"
        )
        self._state = InitState.INITIALIZING

    def _leave(self, *, succeeded: bool) -> None:
        assert self._state is InitState.INITIALIZING
        self._state = InitState.INITED if succeeded else InitState.INIT_FAILED


def init_once(fn: _T) -> _T:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            with self._init_once.guarding():
                return await fn(self, *args, **kwargs)

        return async_guarded  # type: ignore[return-value]

    @functools.wraps(fn)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._init_once.guarding():
            return fn(self, *args, **kwargs)

    return guarded  # type: ignore[return-value]
