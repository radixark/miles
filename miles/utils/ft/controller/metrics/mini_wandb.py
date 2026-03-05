import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from miles.utils.ft.protocols.metrics import StepValue, TimedStepValue

logger = logging.getLogger(__name__)


@dataclass
class _StepRecord:
    step: int
    receive_time: datetime
    metrics: dict[str, float]


class MiniWandb:
    def __init__(
        self,
        active_run_id: str | None = None,
        max_steps: int = 10000,
        max_age: timedelta = timedelta(minutes=60),
    ) -> None:
        self._active_run_id = active_run_id
        self._max_steps = max_steps
        self._max_age = max_age
        self._data: deque[_StepRecord] = deque()
        self._last_step: int = -1

    def set_active_run_id(self, run_id: str) -> None:
        if run_id != self._active_run_id:
            self._last_step = -1
        self._active_run_id = run_id

    def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
        receive_time: datetime | None = None,
    ) -> None:
        if self._active_run_id is not None and run_id != self._active_run_id:
            logger.debug(
                "Discarding log_step: run_id=%s does not match active=%s",
                run_id,
                self._active_run_id,
            )
            return

        if step <= self._last_step:
            logger.debug(
                "Discarding log_step: step=%d <= last_step=%d",
                step,
                self._last_step,
            )
            return

        self._last_step = step

        record = _StepRecord(
            step=step,
            receive_time=receive_time or datetime.now(timezone.utc),
            metrics=metrics,
        )

        self._data.append(record)
        self._evict()

    def query_last_n_steps(
        self,
        metric_name: str,
        last_n: int,
    ) -> list[StepValue]:
        snapshot = list(self._data)
        result: list[StepValue] = []
        for record in reversed(snapshot):
            if metric_name in record.metrics:
                result.append(StepValue(step=record.step, value=record.metrics[metric_name]))
                if len(result) >= last_n:
                    break

        result.reverse()
        return result

    def query_time_window(
        self,
        metric_name: str,
        window: timedelta,
    ) -> list[TimedStepValue]:
        cutoff = datetime.now(timezone.utc) - window
        snapshot = list(self._data)
        result: list[TimedStepValue] = []
        for record in snapshot:
            if record.receive_time >= cutoff and metric_name in record.metrics:
                result.append(TimedStepValue(
                    step=record.step,
                    timestamp=record.receive_time,
                    value=record.metrics[metric_name],
                ))

        return result

    def latest(self, metric_name: str) -> float | None:
        snapshot = list(self._data)
        for record in reversed(snapshot):
            if metric_name in record.metrics:
                return record.metrics[metric_name]

        return None

    def clear(self) -> None:
        self._data.clear()
        self._last_step = -1

    def _evict(self) -> None:
        while len(self._data) > self._max_steps:
            self._data.popleft()

        cutoff = datetime.now(timezone.utc) - self._max_age
        while self._data and self._data[0].receive_time < cutoff:
            self._data.popleft()
