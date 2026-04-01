import asyncio
import threading
from collections.abc import Coroutine
from typing import TypeVar

__all__ = ["get_async_loop", "run", "eager_create_task"]

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

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


async def eager_create_task(coro: Coroutine[object, object, _T]) -> asyncio.Task[_T]:
    """Create a task and yield so it starts executing immediately.

    Unlike bare ``asyncio.create_task``, this ensures the task's first code
    (including any ``.remote()`` calls) runs before the caller continues.
    Needed when the dispatched task and the caller participate in the same
    distributed collective (e.g. ``sync_actor_critic_data``).
    """
    task = asyncio.create_task(coro)
    await asyncio.sleep(0)
    return task
