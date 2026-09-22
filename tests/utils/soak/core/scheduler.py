import random
from dataclasses import dataclass

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.events import SoakEvent, SoakScheduleEvent
from tests.utils.soak.core.types import SoakActionRequest, SoakForms

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


@dataclass(frozen=True, kw_only=True)
class SoakActionScheduler:
    rng: random.Random
    mean_intervals: dict[str, float]
    forms: SoakForms
    config: SoakRunnerConfig
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED

    def initial_schedule(self) -> SoakScheduleEvent:
        raise NotImplementedError

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        raise NotImplementedError
