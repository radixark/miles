from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
import sys
from argparse import Namespace
from types import ModuleType

import pytest

if "sglang_router" not in sys.modules:
    sglang_router_stub = ModuleType("sglang_router")
    sglang_router_stub.__version__ = "0.0.0"
    sys.modules["sglang_router"] = sglang_router_stub

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.utils.types import Sample


class FakeGenerateState:
    def __init__(self, args: Namespace):
        self.args = args
        self.sampling_params = {}
        self.aborted = False
        self.reset_count = 0

    def reset(self) -> None:
        self.aborted = False
        self.reset_count += 1


def make_group(group_index: int, group_size: int) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 100 + sample_index,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=Sample.Status.COMPLETED,
        )
        for sample_index in range(group_size)
    ]


@pytest.mark.asyncio
async def test_generate_rollout_without_backfill_flag_uses_legacy_group_scheduler(monkeypatch):
    group = make_group(group_index=1, group_size=2)
    args = Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        dynamic_sampling_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    state = FakeGenerateState(args)
    called = []

    async def noop_configure_sglang(_args):
        return None

    async def noop_recompute(*_args, **_kwargs):
        return None

    async def fake_group_level(*_args, **_kwargs):
        called.append("group_level")
        return [group], [group], []

    async def unexpected_sample_completion_backfill(*_args, **_kwargs):
        raise AssertionError("sample-completion backfill should be disabled by default")

    monkeypatch.setattr(train.dumper_utils, "configure_sglang", noop_configure_sglang)
    monkeypatch.setattr(train, "recompute_samples_rollout_logprobs_via_prefill", noop_recompute)
    monkeypatch.setattr(train, "load_function", lambda _path: None)
    monkeypatch.setattr(train, "_generate_rollout_group_level_async", fake_group_level)
    monkeypatch.setattr(
        train,
        "_generate_rollout_sample_completion_backfill_async",
        unexpected_sample_completion_backfill,
    )

    output, aborted_samples = await train.generate_rollout_async(state, rollout_id=0, data_source=lambda _n: [])

    assert called == ["group_level"]
    assert output.samples == [group]
    assert aborted_samples == []
    assert state.reset_count == 1


@pytest.mark.asyncio
async def test_sample_completion_backfill_submits_group_after_enough_samples_finish(monkeypatch):
    args = Namespace(rollout_batch_size=1, n_samples_per_prompt=2)
    state = FakeGenerateState(args)
    data_source_calls = []
    submitted_group_indices = []
    next_group_index = 0

    def data_source(num_groups: int) -> list[list[Sample]]:
        nonlocal next_group_index
        data_source_calls.append(num_groups)
        groups = []
        for _ in range(num_groups):
            next_group_index += 1
            groups.append(make_group(group_index=next_group_index, group_size=args.n_samples_per_prompt))
        return groups

    async def never_complete():
        await asyncio.Future()

    async def complete_group(group: list[Sample]) -> list[Sample]:
        await asyncio.sleep(0)
        return group

    def fake_submit_generate_tasks(_state, samples, sample_done_callback=None):
        tasks = []
        for group in samples:
            submitted_group_indices.append(group[0].group_index)
            if len(submitted_group_indices) == 1:
                assert sample_done_callback is not None
                for _ in group:
                    sample_done_callback()
                tasks.append(asyncio.create_task(never_complete()))
            else:
                tasks.append(asyncio.create_task(complete_group(group)))
        return tasks

    async def fake_abort(_state, pendings, _rollout_id):
        _state.aborted = True
        for task in pendings:
            task.cancel()
        await asyncio.gather(*pendings, return_exceptions=True)
        return []

    monkeypatch.setattr(train, "submit_generate_tasks", fake_submit_generate_tasks)
    monkeypatch.setattr(train, "abort", fake_abort)

    data, all_data, aborted_samples = await train._generate_rollout_sample_completion_backfill_async(
        state,
        rollout_id=0,
        data_source=data_source,
        dynamic_filter=None,
        metric_gatherer=MetricGatherer(),
    )

    assert data_source_calls == [1, 1]
    assert submitted_group_indices == [1, 2]
    assert data == [make_group(group_index=2, group_size=args.n_samples_per_prompt)]
    assert all_data == data
    assert aborted_samples == []
