import asyncio
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.rollout import sglang_rollout
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import ExplicitlyDroppedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


class _GenerateState:
    def __init__(self, groups: list[list]) -> None:
        self._groups = iter(groups)
        self.remaining_batch_size = 0
        self.pendings: set[asyncio.Future] = set()
        self.sampling_params = {}

    def submit_generate_tasks(self, groups: list[list]) -> None:
        for _ in groups:
            group = next(self._groups)
            future = asyncio.get_running_loop().create_future()
            future.set_result(group)
            self.pendings.add(future)
            self.remaining_batch_size += 1

    def reset(self) -> None:
        pass


class _LegacyGenerateState:
    def __init__(self) -> None:
        self.aborted = False
        self.semaphore = asyncio.Semaphore(1)

    def dp_rank_context(self):
        return nullcontext()


async def _no_op(*args, **kwargs) -> None:
    pass


async def _empty_list(*args, **kwargs) -> list:
    return []


@pytest.fixture(autouse=True)
def _reset_event_logger() -> Iterator[None]:
    yield
    set_event_logger(None)


def _drop_events(event_dir: Path) -> list[ExplicitlyDroppedSamplesEvent]:
    return [event for event in read_events(event_dir) if isinstance(event, ExplicitlyDroppedSamplesEvent)]


class TestLegacyRolloutSampleOwnership:
    async def test_generate_stamps_compact_rows_with_the_issued_source(self, monkeypatch) -> None:
        """The legacy generate boundary preserves one issued source across compact rows."""
        source = make_sample(index=7, reward=0.0)
        rows = [make_sample(index=7, reward=0.0), make_sample(index=7, reward=0.0)]
        args = make_args(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=True,
            custom_generate_function_path="compact",
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _LegacyGenerateState())

        async def generate(_input):
            return SimpleNamespace(samples=rows)

        monkeypatch.setattr(sglang_rollout, "load_generate_function", lambda _path: generate)

        output = await sglang_rollout.generate_and_rm(args, source, {}, evaluation=False)

        assert [(row.source_sample_index, row.sample_row_index, row.sample_row_count) for row in output] == [
            (7, 0, 2),
            (7, 1, 2),
        ]

    @pytest.mark.parametrize("partial_rollout", [False, True])
    async def test_abort_resolves_only_groups_that_will_not_retry(
        self, monkeypatch, tmp_path: Path, partial_rollout: bool
    ) -> None:
        """Abort records terminal groups while preserving partial groups for retry."""
        group = [make_sample(index=7, reward=0.0)]
        task = asyncio.get_running_loop().create_future()
        task.set_result(group)
        state = SimpleNamespace(aborted=False, pendings={task})
        args = make_args(
            partial_rollout=partial_rollout,
            use_miles_router=True,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "get", lambda _url: _return({"urls": []}))
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", _no_op)
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))

        recovered = await sglang_rollout.abort(args, rollout_id=4)

        events = _drop_events(tmp_path)
        if partial_rollout:
            assert recovered == [group]
            assert events == []
        else:
            assert recovered == []
            [event] = events
            assert event.sample_indices == [7]
            assert event.reason == "abort"

    async def test_oversampling_records_the_finished_group_not_selected(self, monkeypatch, tmp_path: Path) -> None:
        """The legacy oversampling branch resolves a completed group beyond the target."""
        groups = [[make_sample(index=0, reward=0.0)], [make_sample(index=1, reward=1.0)]]
        args = make_args(
            rollout_batch_size=1,
            over_sampling_batch_size=2,
            n_samples_per_prompt=1,
            dynamic_sampling_filter_path=None,
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
            partial_rollout=False,
            reward_key=None,
        )
        state = _GenerateState(groups)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: None)
        monkeypatch.setattr(sglang_rollout.dumper_utils, "configure_sglang", _no_op)
        monkeypatch.setattr(sglang_rollout, "abort", _empty_list)
        monkeypatch.setattr(sglang_rollout, "recompute_samples_rollout_logprobs_via_prefill", _no_op)
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))

        output, _ = await sglang_rollout.generate_rollout_async(args, 4, lambda count: [None] * count)

        [event] = _drop_events(tmp_path)
        assert event.reason == "oversampling"
        assert {output.samples[0][0].index, event.sample_indices[0]} == {0, 1}

    async def test_dynamic_filter_records_the_rejected_source(self, monkeypatch, tmp_path: Path) -> None:
        """The legacy dynamic-filter branch resolves the rejected source sample."""
        groups = [[make_sample(index=0, reward=0.0)], [make_sample(index=1, reward=1.0)]]
        args = make_args(
            rollout_batch_size=1,
            over_sampling_batch_size=1,
            n_samples_per_prompt=1,
            dynamic_sampling_filter_path="filter",
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
            partial_rollout=False,
            reward_key=None,
        )
        state = _GenerateState(groups)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(
            sglang_rollout,
            "load_function",
            lambda _path: lambda _args, group: DynamicFilterOutput(keep=group[0].index == 1, reason="low"),
        )
        monkeypatch.setattr(sglang_rollout.dumper_utils, "configure_sglang", _no_op)
        monkeypatch.setattr(sglang_rollout, "abort", _empty_list)
        monkeypatch.setattr(sglang_rollout, "recompute_samples_rollout_logprobs_via_prefill", _no_op)
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))

        await sglang_rollout.generate_rollout_async(args, 4, lambda count: [None] * count)

        [event] = _drop_events(tmp_path)
        assert event.sample_indices == [0]
        assert event.reason == "dynamic_filter"

    async def test_sample_filter_records_removed_sources(self, monkeypatch, tmp_path: Path) -> None:
        """The legacy final sample filter resolves every group it removes in place."""
        groups = [[make_sample(index=0, reward=0.0)], [make_sample(index=1, reward=1.0)]]
        args = make_args(
            rollout_batch_size=2,
            over_sampling_batch_size=2,
            n_samples_per_prompt=1,
            dynamic_sampling_filter_path=None,
            rollout_sample_filter_path="filter",
            rollout_all_samples_process_path=None,
            partial_rollout=False,
            reward_key=None,
        )
        state = _GenerateState(groups)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: lambda _args, data: data.pop())
        monkeypatch.setattr(sglang_rollout.dumper_utils, "configure_sglang", _no_op)
        monkeypatch.setattr(sglang_rollout, "abort", _empty_list)
        monkeypatch.setattr(sglang_rollout, "recompute_samples_rollout_logprobs_via_prefill", _no_op)
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))

        await sglang_rollout.generate_rollout_async(args, 4, lambda count: [None] * count)

        [event] = _drop_events(tmp_path)
        assert event.sample_indices == [1]
        assert event.reason == "sample_filter"


async def _return(value):
    return value
