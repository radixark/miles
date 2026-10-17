import random
import time
from dataclasses import dataclass, field

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakEvent
from tests.utils.soak.core.types import SoakActionRequest, SoakForms, SoakTarget
from tests.utils.soak.core.views import admission_closed, latest_observation


@dataclass(frozen=True, kw_only=True)
class SoakActionScheduler:
    forms: SoakForms
    config: SoakRunnerConfig
    rng: random.Random = field(init=False)
    due_of_type: dict[str, float] = field(init=False)
    awaiting_applied: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng", random.Random(self.config.seed))
        now = time.monotonic()
        object.__setattr__(
            self,
            "due_of_type",
            {kind: self._draw_due_at(kind, now=now) for kind in sorted(self.config.target_configs)},
        )

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        self._redraw_applied(events=events, now=now)

        if admission_closed(events) is not None:
            return None

        observation = latest_observation(events)
        if observation is None or observation.errors:
            return None
        targets_of_type: dict[str, list[SoakTarget]] = {
            kind: [target for target in observation.targets or [] if target.kind == kind]
            for kind in self.config.target_configs
        }
        due_types = sorted(kind for kind, due_at in self.due_of_type.items() if now >= due_at)
        if not due_types:
            return None

        kind = self.rng.choice(due_types)
        form = self.rng.choice(self.forms[kind])
        targets = targets_of_type[kind]
        if not targets:
            return None
        target = self.rng.choice(targets)
        request = form.maybe_create_request(target=target, observation=observation, events=events, rng=self.rng)
        if request is None:
            return None

        self.awaiting_applied[kind] = request.request_id
        return request

    def _redraw_applied(self, *, events: list[SoakEvent], now: float) -> None:
        applied = {event.request_id for event in events if isinstance(event, SoakActionAppliedEvent)}
        for kind, request_id in list(self.awaiting_applied.items()):
            if request_id in applied:
                self.due_of_type[kind] = self._draw_due_at(kind, now=now)
                del self.awaiting_applied[kind]

    def _draw_due_at(self, kind: str, *, now: float) -> float:
        return now + self.rng.expovariate(1.0 / self.config.target_configs[kind].mean_interval_seconds)
