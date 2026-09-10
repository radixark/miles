import asyncio

from miles.rollout.data_source import DataSource
from miles.rollout.fully_async_data_buffer import (
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    PutOutcomes,
)
from miles.utils.types import Sample


class ReadOnlyDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass


class _AcknowledgingBuffer(DefaultDataBuffer):
    def __init__(self, input: DataBufferConstructorInput) -> None:
        super().__init__(input)
        self.stored = asyncio.Event()
        self.acknowledged = asyncio.Event()

    async def put(self, input: DataBufferInput) -> PutOutcomes:
        outcomes = await super().put(input)
        self.stored.set()
        await self.acknowledged.wait()
        return outcomes
