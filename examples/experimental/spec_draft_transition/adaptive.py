"""Prespecified, held-out-L1 driven one-way loss transition."""

import math
from dataclasses import dataclass, field


@dataclass
class PlateauTransition:
    interval: int = 100
    minimum_updates: int = 500
    window: int = 300
    relative_threshold: float = 0.005
    history: list[tuple[int, float]] = field(default_factory=list)
    switch_step: int | None = None

    def observe(self, step: int, l1: float, allow_switch: bool = True) -> dict:
        if step < 0 or step % self.interval or not math.isfinite(l1) or l1 <= 0:
            raise ValueError((step, l1))
        if self.history and step <= self.history[-1][0]:
            raise ValueError('Observations must increase in step')
        self.history.append((step, l1))
        earlier = [value for update, value in self.history if update <= step - self.window]
        best = min(value for _, value in self.history)
        gain = None if not earlier else (min(earlier) - best) / min(earlier)
        triggered = allow_switch and self.switch_step is None and step >= self.minimum_updates and gain is not None and gain < self.relative_threshold
        if triggered:
            self.switch_step = step
        return dict(step=step, validation_l1=l1, best_l1=best,
                    window_relative_best_improvement=gain, triggered=triggered,
                    switch_step=self.switch_step,
                    rule='step>=500 and relative improvement in best held-out L1 over last 300 updates <0.005')

    def weights(self) -> tuple[float, float, float]:
        """Hard CE, soft CE and sum-over-vocabulary L1 coefficients."""
        return (0.1, 0.9, 0.0) if self.switch_step is None else (0.0, 0.1, 0.9)
