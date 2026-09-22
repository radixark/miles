import random
import time
from dataclasses import dataclass

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.events import SoakEvent, SoakScheduleEvent
from tests.utils.soak.core.types import SoakActionRequest, SoakForms, SoakTarget
from tests.utils.soak.core.views import admission_closed, due_of_type, latest_observation

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


@dataclass(frozen=True, kw_only=True)
class SoakActionScheduler:
    rng: random.Random
    mean_intervals: dict[str, float]
    forms: SoakForms
    config: SoakRunnerConfig
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED

    def __post_init__(self) -> None:
        if set(self.config.target_policies) != set(self.mean_intervals):
            raise ValueError("Target policies must name exactly the scheduled target kinds")

    def initial_schedule(self) -> SoakScheduleEvent:
        return SoakScheduleEvent(
            due_of_type={
                kind: time.monotonic() + self.rng.expovariate(1.0 / mean_interval_seconds)
                for kind, mean_interval_seconds in sorted(self.mean_intervals.items())
            }
        )

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        if admission_closed(events) is not None:
            return None

        observation = latest_observation(events)
        if observation is None or observation.errors:
            return None
        targets_of_type: dict[str, list[SoakTarget]] = {
            kind: [target for target in observation.targets or [] if target.kind == kind]
            for kind in self.mean_intervals
        }
        due_types = sorted(kind for kind, due_at in due_of_type(events).items() if now >= due_at)
        ready_types = [kind for kind in due_types if targets_of_type[kind]]
        if not ready_types:
            return None

        kind = self.rng.choice(ready_types)
        form = self.rng.choice(self.forms[kind])
        target = self.rng.choice(targets_of_type[kind])
        request = form.maybe_create_request(target=target, observation=observation, events=events, rng=self.rng)
        if request is None:
            return None
        return request.model_copy(update={"next_due_at": now + self.rng.expovariate(1.0 / self.mean_intervals[kind])})
