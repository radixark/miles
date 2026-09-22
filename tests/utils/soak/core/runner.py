from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.scheduler import POLL_INTERVAL_SECONDS, SoakActionScheduler
from tests.utils.soak.core.types import SoakForms, SoakObserver

SHUTDOWN_TIMEOUT_SECONDS: float = 180.0


@dataclass(kw_only=True)
class SoakRunner:
    observer: SoakObserver
    scheduler: SoakActionScheduler
    forms: SoakForms
    event_log: EventLog
    config: SoakRunnerConfig
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS

    async def run(self, training: Coroutine[Any, Any, Any], *, teardown: Callable[[], Awaitable[None]]) -> None:
        raise NotImplementedError

    async def finish(self) -> None:
        raise NotImplementedError
