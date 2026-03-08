from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from miles.utils.ft.controller.types import TrainingMetricStoreProtocol


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float

logger = logging.getLogger(__name__)


class MiniWandb(TrainingMetricStoreProtocol):
    def __init__(
        self,
        active_run_id: str | None = None,
        max_steps: int = 10000,
        max_age: timedelta = timedelta(minutes=60),
    ) -> None:
        self._active_run_id = active_run_id
        self._max_steps = max_steps
        self._max_age = max_age
        self._runs: dict[str, deque[_StepRecord]] = {}
        self._last_step: dict[str, int] = {}

    def set_active_run_id(self, run_id: str) -> None:
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

        last = self._last_step.get(run_id, -1)
        if step <= last:
            logger.debug(
                "Discarding log_step: step=%d <= last_step=%d",
                step,
                last,
            )
            return

        self._last_step[run_id] = step

        record = _StepRecord(
            step=step,
            receive_time=receive_time or datetime.now(timezone.utc),
            metrics=metrics,
        )

        data = self._runs.setdefault(run_id, deque())
        data.append(record)
        self._evict(run_id)

    def query_last_n_steps(
        self,
        metric_name: str,
        last_n: int,
    ) -> list[StepValue]:
        snapshot = list(self._active_data())
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
        snapshot = list(self._active_data())
        result: list[TimedStepValue] = []
        for record in snapshot:
            if record.receive_time >= cutoff and metric_name in record.metrics:
                result.append(
                    TimedStepValue(
                        step=record.step,
                        timestamp=record.receive_time,
                        value=record.metrics[metric_name],
                    )
                )

        return result

    def latest(self, metric_name: str) -> float | None:
        snapshot = list(self._active_data())
        for record in reversed(snapshot):
            if metric_name in record.metrics:
                return record.metrics[metric_name]

        return None

    def _active_data(self) -> deque[_StepRecord]:
        if self._active_run_id is None:
            return deque()
        return self._runs.get(self._active_run_id, deque())

    def _evict(self, run_id: str) -> None:
        data = self._runs.get(run_id)
        if data is None:
            return

        while len(data) > self._max_steps:
            data.popleft()

        cutoff = datetime.now(timezone.utc) - self._max_age
        while data and data[0].receive_time < cutoff:
            data.popleft()

        if not data:
            del self._runs[run_id]
            self._last_step.pop(run_id, None)


@dataclass
class _StepRecord:
    step: int
    receive_time: datetime
    metrics: dict[str, float]
