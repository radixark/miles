"""Opt-in cancellation provenance without logging coroutine arguments or locals."""

import asyncio
import contextvars
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)
_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar("async_diagnostic_scope", default=None)


def _emit(event: str, **fields: Any) -> None:
    logger.warning("async_diagnostic %s", json.dumps({"event": event, "pid": os.getpid(), **fields}))


def _location(coro: Any) -> dict[str, Any]:
    code = getattr(coro, "cr_code", None)
    return {} if code is None else {"file": code.co_filename, "line": code.co_firstlineno, "function": code.co_name}


def log_unawaited_coroutine(coro: Any, *, policy: str) -> None:
    # Do not stringify the warning: it may include source lines or arguments.
    _emit(
        "unawaited_coroutine",
        policy=policy,
        coroutine=_location(coro),
        origin=getattr(coro, "cr_origin", None),
        note="origin is thread-local; absence does not mean the coroutine was created on this thread",
    )


class _DiagnosticTask(asyncio.Task):
    def __init__(self, coro: Any, *, loop: asyncio.AbstractEventLoop, **kwargs: Any) -> None:
        context = kwargs.get("context")
        self._diagnostic_scope = context.get(_scope) if context is not None else _scope.get()
        self._created = time.monotonic()
        super().__init__(coro, loop=loop, **kwargs)

    def cancel(self, msg: Any = None) -> bool:
        caller = asyncio.current_task(loop=self.get_loop())
        accepted = super().cancel(msg)
        _emit(
            "task_cancel",
            task_id=id(self),
            caller_task_id=id(caller) if caller is not None else None,
            scope=self._diagnostic_scope,
            accepted=accepted,
            age_s=round(time.monotonic() - self._created, 3),
            coroutine=_location(self.get_coro()),
            # No source text, frame locals, task names or cancellation message.
            caller_stack=[(f.filename, f.lineno, f.name) for f in traceback.extract_stack(limit=16)[:-1]],
        )
        return accepted


def _task_factory(loop: asyncio.AbstractEventLoop, coro: Any, **kwargs: Any) -> asyncio.Task:
    return _DiagnosticTask(coro, loop=loop, **kwargs)


def configure_async_diagnostics(loop: asyncio.AbstractEventLoop) -> None:
    """Install in the loop's owning thread; do not replace an existing task factory."""
    if os.environ.get("MILES_ASYNC_DIAGNOSTICS") != "1":
        return
    factory = loop.get_task_factory()
    if factory is _task_factory:
        return
    if factory is not None:
        _emit("task_factory_conflict", note="cancellation tracing not installed")
        return
    sys.set_coroutine_origin_tracking_depth(8)
    loop.set_task_factory(_task_factory)
    _emit("enabled", loop_type=type(loop).__name__)


@contextmanager
def async_diagnostic_scope(scope: str, *, timeout_s: float) -> Iterator[None]:
    """Associate child tasks with an opaque trial identifier (never a prompt)."""
    if os.environ.get("MILES_ASYNC_DIAGNOSTICS") != "1":
        yield
        return
    configure_async_diagnostics(asyncio.get_running_loop())
    token = _scope.set(scope)
    task = asyncio.current_task()
    previous_scope = task._diagnostic_scope if isinstance(task, _DiagnosticTask) else None
    if isinstance(task, _DiagnosticTask):
        task._diagnostic_scope = scope
    _emit("scope_start", scope=scope, task_id=id(task), timeout_s=timeout_s)
    try:
        yield
    finally:
        _emit("scope_end", scope=scope)
        if isinstance(task, _DiagnosticTask):
            task._diagnostic_scope = previous_scope
        _scope.reset(token)
