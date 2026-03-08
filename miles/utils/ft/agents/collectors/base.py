import asyncio
from abc import ABC, abstractmethod

from miles.utils.ft.models.metrics import CollectorOutput, MetricSample

_COLLECT_TIMEOUT_MULTIPLIER = 2.0


class BaseCollector(ABC):
    collect_interval: float = 10.0

    async def collect(self) -> CollectorOutput:
        timeout = self.collect_interval * _COLLECT_TIMEOUT_MULTIPLIER
        metrics = await asyncio.wait_for(
            asyncio.to_thread(self._collect_sync),
            timeout=timeout,
        )
        return CollectorOutput(metrics=metrics)

    @abstractmethod
    def _collect_sync(self) -> list[MetricSample]: ...

    async def close(self) -> None:  # noqa: B027
        pass
