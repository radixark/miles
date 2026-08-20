import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


__all__ = [
    "get_async_loop",
    "run",
    "submit",
    "wait_futures",
    "wait_cancelling_pending_on_first_completion",
    "eager_create_task",
    "gather_and_raise_first",
]

_T = TypeVar("_T")


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        assert (
            threading.current_thread() is not self._thread
        ), "submitting from the loop thread and then blocking on the result would deadlock the loop"
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        # Schedule a coroutine onto the loop and block until it's done
        return self.submit(coro).result()


# Create one global instance
async_loop = None
_async_loop_lock = threading.Lock()


def get_async_loop():
    global async_loop
    with _async_loop_lock:
        if async_loop is None:
            async_loop = AsyncLoopThread()
        return async_loop


# TODO: rename these functions and classes
def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


def submit(coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
    """Fire a coroutine on the background event loop and return its future."""
    return get_async_loop().submit(coro)


def wait_futures(futures: Sequence[concurrent.futures.Future]) -> list[Any]:
    """Collect a fan-out, raising the first error once every future has settled."""
    results: list[Any] = []
    errors: list[Exception] = []
    for index, future in enumerate(futures):
        try:
            results.append(future.result())
        except Exception as e:
            logger.warning(f"wait_futures index={index} failed", exc_info=e)
            results.append(None)
            errors.append(e)

    if errors:
        raise errors[0]
    return results


async def wait_cancelling_pending_on_first_completion(tasks: Sequence[asyncio.Task]) -> None:
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    errors = [error for task in tasks if (error := _compute_task_error(task)) is not None]
    for error in errors:
        logger.error("task failed", exc_info=error)
    if errors:
        raise errors[0]


def _compute_task_error(task: asyncio.Task) -> BaseException | None:
    if task.cancelled():
        return None
    return task.exception()


async def eager_create_task(coro: Coroutine[object, object, _T]) -> asyncio.Task[_T]:
    """Create a task and yield so it starts executing immediately.

    Unlike bare ``asyncio.create_task``, this ensures the task's first code
    (including any ``.remote()`` calls) runs before the caller continues.
    """
    task = asyncio.create_task(coro)
    await asyncio.sleep(0)
    return task


class AsyncioGatherUtils:
    @staticmethod
    def has_error(outputs):
        return any(isinstance(output, BaseException) for output in outputs)

    @staticmethod
    def log_error(
        outputs,
        debug_name: str = "",
        *,
        describe_failure: Callable[[int], str] | None = None,
        log: Callable[..., None] = logger.warning,
    ) -> None:
        for i, output in enumerate(outputs):
            if isinstance(output, BaseException):
                message = f"{debug_name} error index={i}" if describe_failure is None else describe_failure(i)
                log(message, exc_info=output)


async def gather_and_raise_first(
    awaitables: Sequence[Awaitable[_T]], *, describe_failure: Callable[[int], str] | None = None
) -> list[_T]:
    results = await asyncio.gather(*awaitables, return_exceptions=True)

    if describe_failure is not None:
        AsyncioGatherUtils.log_error(results, describe_failure=describe_failure, log=logger.error)

    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise failures[0]
    return results
