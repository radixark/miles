import asyncio
import logging
from dataclasses import dataclass, field

from miles.rollout.session.errors import SessionServerClosingError

logger = logging.getLogger(__name__)


@dataclass
class SessionActivity:
    generation: int = 0
    active: int = 0
    retaining: bool = False
    timer: asyncio.TimerHandle | None = field(default=None, repr=False)


class SessionLifecycle:
    """Track admitted serving work and collect retention on one event loop."""

    def __init__(self, registry, *, idle_timeout: float = 300.0):
        self.registry = registry
        self.idle_timeout = idle_timeout
        self.closing = False
        self._shutdown: asyncio.Task | None = None
        self.tasks: set[asyncio.Task] = set()

    def check_open(self) -> None:
        if self.closing:
            raise SessionServerClosingError("Session server is shutting down")

    def accept(self, session, *, collect: bool = False) -> None:
        self.check_open()
        activity = session.activity
        activity.generation += 1
        activity.active += 1
        activity.retaining |= collect
        self.forget_timer(session)

    def finish(self, session) -> None:
        # No await: even a repeatedly cancelled request must retire its activity.
        session.activity.active -= 1
        if session.activity.retaining and not session.activity.active and not session.closing and not self.closing:
            self._arm(session)

    def forget_timer(self, session) -> None:
        if session.activity.timer is not None:
            session.activity.timer.cancel()
            session.activity.timer = None

    def _arm(self, session) -> None:
        self.forget_timer(session)

        def expire():
            task = asyncio.create_task(self._expire(session, handle))
            self.tasks.add(task)
            task.add_done_callback(self._finished)

        handle = asyncio.get_running_loop().call_later(self.idle_timeout, expire)
        session.activity.timer = handle

    async def _expire(self, session, handle: asyncio.TimerHandle) -> None:
        async with session.lock:
            if (
                self.closing
                or session.closing
                or self.registry.sessions.get(session.session_id) is not session
                or session.activity.timer is not handle
                or session.activity.active
            ):
                return
            session.closing = True
            self.forget_timer(session)
            self.registry.remove_session(session.session_id)

    def _finished(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Session retention cleanup failed", exc_info=task.exception())

    def stop(self) -> None:
        self.closing = True
        for session in self.registry.sessions.values():
            session.closing = True
            self.forget_timer(session)

    async def close(self, *, timeout: float = 30.0) -> bool:
        if self._shutdown is None:
            self.stop()
            self._shutdown = asyncio.create_task(self._close())
        try:
            return await asyncio.wait_for(asyncio.shield(self._shutdown), timeout)
        except TimeoutError:
            logger.warning("Session server shutdown timed out; outstanding work retains its storage")
            return False

    async def _close(self) -> bool:
        for session in list(self.registry.sessions.values()):
            async with session.lock:
                self.registry.remove_session(session.session_id)
        if self.tasks:
            await asyncio.gather(*self.tasks)
        return await self.registry.record_store.close(timeout=None)
