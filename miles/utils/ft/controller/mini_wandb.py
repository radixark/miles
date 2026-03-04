import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from miles.utils.ft.models import StepValue, TimedStepValue

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
        self._data: dict[int, deque[_StepRecord]] = {}

    def set_active_run_id(self, run_id: str) -> None:
        self._active_run_id = run_id

    def log_step(
        self,
        run_id: str,
        rank: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        if self._active_run_id is not None and run_id != self._active_run_id:
            logger.debug(
                "Discarding log_step: run_id=%s does not match active=%s",
                run_id,
                self._active_run_id,
            )
            return

        record = _StepRecord(
            step=step,
            receive_time=datetime.now(timezone.utc),
            metrics=metrics,
        )

        if rank not in self._data:
            self._data[rank] = deque()
        self._data[rank].append(record)

        self._evict(rank)

    def query_last_n_steps(
        self,
        metric_name: str,
        rank: int,
        last_n: int,
    ) -> list[StepValue]:
        if rank not in self._data:
            return []

        result: list[StepValue] = []
        for record in reversed(self._data[rank]):
            if metric_name in record.metrics:
                result.append(StepValue(step=record.step, value=record.metrics[metric_name]))
                if len(result) >= last_n:
                    break

        result.reverse()
        return result

    def query_time_window(
        self,
        metric_name: str,
        rank: int,
        window: timedelta,
    ) -> list[TimedStepValue]:
        if rank not in self._data:
            return []

        cutoff = datetime.now(timezone.utc) - window
        result: list[TimedStepValue] = []
        for record in self._data[rank]:
            if record.receive_time >= cutoff and metric_name in record.metrics:
                result.append(TimedStepValue(
                    step=record.step,
                    timestamp=record.receive_time,
                    value=record.metrics[metric_name],
                ))

        return result

    def latest(self, metric_name: str, rank: int) -> float | None:
        if rank not in self._data:
            return None

        for record in reversed(self._data[rank]):
            if metric_name in record.metrics:
                return record.metrics[metric_name]

        return None

    def clear(self) -> None:
        self._data.clear()

    def _evict(self, rank: int) -> None:
        buffer = self._data[rank]

        while len(buffer) > self._max_steps:
            buffer.popleft()

        cutoff = datetime.now(timezone.utc) - self._max_age
        while buffer and buffer[0].receive_time < cutoff:
            buffer.popleft()
