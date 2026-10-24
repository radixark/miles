"""Opt-in cancellation provenance without logging coroutine arguments or locals."""

import asyncio
import contextvars
import itertools
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
# Process-unique task identifiers: id() is a memory address that CPython reuses
# once a task is collected, so it cannot correlate events across a long run.
_task_ids = itertools.count(1)
_STACK_LIMIT = 16


def _emit(event: str, *, level: int = logging.WARNING, **fields: Any) -> None:
    # Only defects are warnings; lifecycle events stay informational so that
    # WARNING-based alerting is not tripped by every successful trial.
    if not logger.isEnabledFor(level):
        return
    logger.log(level, "async_diagnostic %s", json.dumps({"event": event, "pid": os.getpid(), **fields}))


def _location(coro: Any) -> dict[str, Any]:
    code = getattr(coro, "cr_code", None)
    return {} if code is None else {"file": code.co_filename, "line": code.co_firstlineno, "function": code.co_name}


def _task_id(task: asyncio.Task | None) -> int | None:
    return task._diagnostic_id if isinstance(task, _DiagnosticTask) else None


def _caller_stack(frame: Any) -> list[tuple[str, int | None, str]]:
    # lookup_lines=False skips reading source files: no source text is logged,
    # and cancellation storms must not add file I/O on the event loop.
    summary = traceback.StackSummary.extract(traceback.walk_stack(frame), limit=_STACK_LIMIT, lookup_lines=False)
    return [(f.filename, f.lineno, f.name) for f in reversed(summary)]


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
        self._diagnostic_id = next(_task_ids)
        self._created = time.monotonic()
        super().__init__(coro, loop=loop, **kwargs)

    def cancel(self, msg: Any = None) -> bool:
        caller = asyncio.current_task(loop=self.get_loop())
        accepted = super().cancel(msg)
        if not logger.isEnabledFor(logging.WARNING):
            return accepted
        _emit(
            "task_cancel",
            task_id=self._diagnostic_id,
            caller_task_id=_task_id(caller),
            scope=self._diagnostic_scope,
            accepted=accepted,
            age_s=round(time.monotonic() - self._created, 3),
            coroutine=_location(self.get_coro()),
            # No source text, frame locals, task names or cancellation message.
            caller_stack=_caller_stack(sys._getframe(1)),
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
    _emit("enabled", level=logging.INFO, loop_type=type(loop).__name__)


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
    _emit("scope_start", level=logging.INFO, scope=scope, task_id=_task_id(task), timeout_s=timeout_s)
    try:
        yield
    finally:
        _emit("scope_end", level=logging.INFO, scope=scope)
        if isinstance(task, _DiagnosticTask):
            task._diagnostic_scope = previous_scope
        _scope.reset(token)
