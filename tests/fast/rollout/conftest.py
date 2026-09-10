import asyncio

from miles.rollout.fully_async_data_buffer import (
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    PutOutcomes,
)


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
