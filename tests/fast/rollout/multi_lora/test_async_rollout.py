import asyncio
import threading
from collections import defaultdict
from types import SimpleNamespace

from miles.rollout.multi_lora.async_rollout import AsyncMultiLoRAWorker, GroupBuffer, MultiLoRAWorkerMetrics
from miles.utils.types import AdapterRef, Sample


class AsyncDataSourceFake:
    def __init__(self, group: list[Sample]) -> None:
        self.group = group
        self.completed = False

    async def get_samples(self, num_samples: int) -> list[list[Sample]]:
        await asyncio.sleep(0)
        self.completed = True
        return [self.group]


class TestRunLoop:
    async def test_run_loop_awaits_the_async_data_source_before_processing_a_group(self) -> None:
        """The producer awaits sample retrieval before processing the returned group."""
        group = [Sample(prompt="prompt")]
        data_source = AsyncDataSourceFake(group)
        worker = AsyncMultiLoRAWorker.__new__(AsyncMultiLoRAWorker)

        async def generate(args: SimpleNamespace, received: list[Sample], sampling_params: dict) -> list[Sample]:
            assert data_source.completed
            received[0].adapter = AdapterRef(name="adapter", slot=1)
            received[0].response = "generated"
            worker.running = False
            return received

        worker.args = SimpleNamespace()
        worker.data_source = data_source
        worker.generate_fn = generate
        worker.concurrency = 1
        worker.running = True
        worker.failure = None
        worker.state = SimpleNamespace(sampling_params={})
        worker.dynamic_filter = None
        worker.buffer_lock = threading.Lock()
        worker.buffers = defaultdict(GroupBuffer)
        worker.metrics = MultiLoRAWorkerMetrics()

        await asyncio.wait_for(worker.run_loop(), timeout=1)

        assert worker.buffers["adapter"].get(1) == [group]
        assert group[0].response == "generated"
        assert worker.failure is None
