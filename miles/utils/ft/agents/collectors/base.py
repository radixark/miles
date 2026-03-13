import asyncio
from abc import ABC, abstractmethod

from miles.utils.ft.agents.types import MetricSample
from miles.utils.ft.utils.base_model import FtBaseModel

_COLLECT_TIMEOUT_MULTIPLIER = 2.0


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]


class BaseCollector(ABC):
    collect_interval: float = 10.0

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "collect_interval" in cls.__dict__:
            val = cls.__dict__["collect_interval"]
            if isinstance(val, (int, float)) and val < 0:
                raise ValueError(f"{cls.__name__}.collect_interval must be >= 0, got {val}")

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
