import asyncio
import threading
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

from miles.rollout.multi_lora import async_rollout
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


class _FakeController:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_value = snapshot
        self.records: list[tuple[int, dict[str, int], list[str]]] = []

    async def snapshot(self) -> dict[str, Any]:
        return self.snapshot_value

    async def record_batch_adapters(
        self, rollout_id: int, group_counts: dict[str, int], step_names: list[str]
    ) -> None:
        self.records.append((rollout_id, group_counts, step_names))


class _FakeMetrics:
    def record_shipped_samples(self, _args: Any, _data: Any, _step_names: Any, _adapters: Any) -> dict:
        return {}

    def pop_stale_drops(self) -> dict:
        return {}

    def pop_metrics(self) -> dict:
        return {}


class _FakeWorker:
    def __init__(self, group) -> None:
        self.group = group
        self.failure = None
        self.metrics = _FakeMetrics()
        self.worker_thread = SimpleNamespace(is_alive=lambda: True)

    def queue_sizes(self) -> dict[str, int]:
        return {"alpha": 1}

    def queue_size(self) -> int:
        return 1

    def get_groups(self, _snapshot, _capacity, group_counts):
        if self.group is None:
            return [], group_counts
        group = self.group
        self.group = None
        return [group], {"alpha": 1}


class TestGenerateRolloutMultiLoRAAsync:
    async def test_generate_records_the_collected_batch_on_the_independent_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generation awaits an independent-controller snapshot and records the collected batch bookkeeping."""
        adapter = SimpleNamespace(
            name="alpha",
            step=4,
            slot=1,
            accumulated_groups=0,
            registration_id="registration",
            config=SimpleNamespace(adapter_global_batch_size=1, rollout_batch_size=1, n_samples_per_prompt=1),
        )
        controller = _FakeController({"active": {"alpha": adapter}, "retiring": {}})
        sample = async_rollout.Sample(
            index=0,
            metadata={"registration_id": "registration"},
            adapter=SimpleNamespace(name="alpha", slot=1),
        )
        worker = _FakeWorker([sample])
        args = SimpleNamespace(
            rollout_global_dataset=True,
            rollout_sample_filter_path=None,
            global_batch_size=1,
            multi_lora_max_coalesce_wait_s=0.01,
            multi_lora_max_empty_wait_s=0.01,
            hf_checkpoint="model",
            chat_template_path=None,
            sglang_server_concurrency=1,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            rollout_temperature=1.0,
            rollout_top_p=1.0,
            rollout_top_k=-1,
            rollout_max_response_len=8,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_skip_special_tokens=False,
            sglang_dp_size=1,
            sglang_model_routers=None,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        monkeypatch.setattr(async_rollout, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr("miles.rollout.sglang_rollout.load_tokenizer", lambda *_args, **_kwargs: object())
        monkeypatch.setattr("miles.rollout.sglang_rollout.load_processor", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(async_rollout, "recompute_samples_rollout_logprobs_via_prefill", _completed_none)
        monkeypatch.setattr(async_rollout.AsyncMultiLoRAWorker, "global_worker", worker)

        output = await async_rollout.generate_rollout_multi_lora_async(args, 9, object())

        assert output.samples == [[sample]]
        assert controller.records == [(9, {"alpha": 1}, ["alpha"])]


async def _completed_none(*_args: Any, **_kwargs: Any) -> None:
    return None
