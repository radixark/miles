from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import argparse
import asyncio
from argparse import Namespace
from collections import deque
from dataclasses import replace

import pytest

import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.inference_rollout.compatibility import call_rollout_function
from miles.utils.async_utils import run
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

N_SAMPLES_PER_PROMPT = 2


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        self.recycled.extend(groups)


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
) -> list[Sample]:
    versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
        for version in weight_versions or []
    ]
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=status,
            weight_versions=list(versions),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        async_max_concurrent_samples=None,
        async_data_buffer_capacity_factor=1000.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_request_timeout_secs=14400,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    return fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_eval_without_fleet_pauses_producer(monkeypatch):
    """Shared-engine eval: producer submissions pause during eval and resume after."""
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch, make_args(rollout_batch_size=2, eval_num_gpus=0), data_source, generate=blocking_generate
    )

    eval_started = asyncio.Event()
    eval_release = asyncio.Event()
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}

    async def fake_run_eval_datasets(state, cache):
        assert state is fn.state  # shared-engine eval uses the train state
        eval_started.set()
        await eval_release.wait()
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    # Start the producer via a train call, then run eval concurrently.
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == submitted_before_eval

    eval_release.set()
    output = await eval_task
    assert output.data == eval_results

    # Producer resumes and the train drain completes.
    assert (await drain).samples


async def test_eval_runs_on_dedicated_fleet(monkeypatch):
    """RolloutManager (not the fn) decides fleet-vs-shared and builds the fleet's
    GenerateState; it hands it in via RolloutFnEvalInput.generate_state. The fn must
    use that state as-is (not self.state) and must not touch the producer/data_source.
    Building/caching the fleet state itself is RolloutExecutorEvalFleet's job, covered in
    tests/fast/rollout/test_checkpoint_eval.py.
    """
    args = make_args(eval_num_gpus=1, eval_num_gpus_per_engine=1)
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, args, data_source)

    fleet_state = FakeGenerateState(args)
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}
    seen_states = []

    async def fake_run_eval_datasets(state, cache):
        seen_states.append(state)
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    output = await fn(RolloutFnEvalInput(rollout_id=0, generate_state=fleet_state, weight_version="0"))

    assert output.data == eval_results
    assert seen_states == [fleet_state]  # used the fleet's state, not fn.state
    # Eval must not start the producer or consume training prompts.
    assert fn._worker is None
    assert data_source.num_get_calls == 0


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1


async def test_stale_group_recycled(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = [
                        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
                        for version in data_source_fresh_versions
                    ]
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    args = make_args(rollout_batch_size=1, max_weight_staleness=2, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 5


async def test_stale_group_dropped_by_default(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_worker_failure_beats_queued_groups(monkeypatch):
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        raise RuntimeError("generation exploded")

    fn._output = make_buffer()[0]
    group = make_group(1)
    await fn._output.put(data_buffer.DataBufferInput(prompt_group=group, group=group))
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_buffer_failure_beats_queued_groups(monkeypatch):
    """A background buffer failure is terminal even when another subgroup is already readable."""
    error = RuntimeError("buffer dispatch exploded")
    entry = data_buffer.DataBufferInput(prompt_group=make_group(1), group=make_group(1))

    class _FailedBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            raise AssertionError("unreachable")

        async def get(self, **context) -> data_buffer.DataBufferInput:
            return entry

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

        async def wait_failed(self) -> None:
            raise error

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
    fn._output = _FailedBuffer()
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(RuntimeError, match="buffer dispatch exploded") as failed:
            await fn(RolloutFnTrainInput(rollout_id=0))
        assert failed.value is error
    finally:
        await fn.aclose()


async def test_a_cancelled_inner_dispatch_fails_the_driver(monkeypatch):
    """An inner cancellation is a terminal run failure, not a clean cancellation of one policy."""
    error = asyncio.CancelledError("custom inner cancelled itself")

    class _CancelledInner(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            raise error

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self) -> dict[str, float]:
            return {}

        async def aclose(self) -> None:
            return None

    buffer, _ = make_multi_buffer("solver", "verifier")
    buffer._inners["verifier"] = _CancelledInner()
    await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
    fn._output = buffer
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(RuntimeError, match="verifier.*cancelled") as failed:
            await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="solver"))
        assert failed.value.__cause__ is error
    finally:
        with pytest.raises(RuntimeError, match="verifier.*cancelled") as closed:
            await fn.aclose()
        assert closed.value.__cause__ is error


async def test_worker_leaves_custom_publish_ordering_to_the_buffer(monkeypatch):
    """The worker forwards completed groups concurrently and leaves a custom buffer to order its own puts."""
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    disjoint_started = asyncio.Event()
    overlapping_started = asyncio.Event()

    class _RouteBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            group_index = data_buffer.first_sample(input.group).group_index
            if group_index == 1:
                first_started.set()
                await first_release.wait()
            elif group_index == 2:
                disjoint_started.set()
            elif group_index == 3:
                overlapping_started.set()

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

        async def aclose(self) -> None:
            first_release.set()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = data_buffer.first_sample(group).group_index
        if group_index == 2:
            await first_started.wait()
        elif group_index == 3:
            await disjoint_started.wait()
        elif group_index > 3:
            await asyncio.Event().wait()
        model_id = "b" if group_index == 2 else "a"
        return make_multi_policy_group(group_index, model_id)

    args = make_args(rollout_batch_size=3, rollout_submission_granularity="group")
    fn = make_fn(
        monkeypatch,
        args,
        FakeDataSource(scripted=[make_group(group_index) for group_index in (1, 2, 3)]),
        generate=generate,
    )
    fn._output = _RouteBuffer()
    fn._worker = asyncio.create_task(fn._worker_loop())
    try:
        await asyncio.wait_for(first_started.wait(), timeout=0.1)
        await asyncio.wait_for(disjoint_started.wait(), timeout=0.1)
        await asyncio.wait_for(overlapping_started.wait(), timeout=0.1)

        first_release.set()
    finally:
        await fn.aclose()


async def _exercise_worker_publish_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    third_model_ids: tuple[str, ...],
) -> None:
    second_generation_release = asyncio.Event()
    third_generation_started = asyncio.Event()
    third_generation_release = asyncio.Event()
    second_publish_started = asyncio.Event()
    third_publish_started = asyncio.Event()

    args = make_args(
        rollout_batch_size=1,
        async_data_buffer_capacity_factor=1.0,
        megatron_config=encode_megatron_config("solver", "verifier"),
    )

    class _ObservedMultiBuffer(data_buffer.DefaultMultiDataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            group_index = data_buffer.first_sample(input.group).group_index
            if group_index == 2:
                second_publish_started.set()
            elif group_index == 3:
                third_publish_started.set()
            await super().put(input)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = data_buffer.first_sample(group).group_index
        for _ in group:
            sample_done_callback()
        if group_index == 2:
            await second_generation_release.wait()
            return make_multi_policy_group(group_index, "solver")
        if group_index == 3:
            third_generation_started.set()
            await third_generation_release.wait()
            return make_multi_policy_group(group_index, *third_model_ids)
        if group_index > 3:
            await asyncio.Event().wait()
        return make_multi_policy_group(group_index, "solver")

    data_source = FakeDataSource(scripted=[make_group(index) for index in (1, 2, 3)])
    fn = make_fn(monkeypatch, args, data_source, generate=generate)
    buffer = _ObservedMultiBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None)
    )
    solver_inner = buffer._inners["solver"]
    await solver_inner.put(data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver")))
    fn._output = buffer
    fn._worker = asyncio.create_task(fn._worker_loop())
    verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
    try:
        await asyncio.wait_for(third_generation_started.wait(), timeout=0.1)
        second_generation_release.set()
        await asyncio.wait_for(second_publish_started.wait(), timeout=0.1)
        third_generation_release.set()
        await asyncio.wait_for(third_publish_started.wait(), timeout=0.1)

        verifier_entry = await asyncio.wait_for(verifier, timeout=0.1)
        assert data_buffer.first_sample(verifier_entry.group).group_index == 3

        expected_solver_indices = [0, 1, 2, 3] if "solver" in third_model_ids else [0, 1, 2]
        solver_entries = [await asyncio.wait_for(solver_inner.get(), timeout=0.1) for _ in expected_solver_indices]
        assert [
            data_buffer.first_sample(entry.group).group_index for entry in solver_entries
        ] == expected_solver_indices
    finally:
        second_generation_release.set()
        third_generation_release.set()
        if not verifier.done():
            verifier.cancel()
        await asyncio.gather(verifier, return_exceptions=True)
        await fn.aclose()


async def test_worker_publishes_an_overlapping_route_before_a_blocked_shared_route(monkeypatch):
    """A blocked solver-only route cannot hide a later verifier sibling that has separate route credit."""
    await _exercise_worker_publish_route(monkeypatch, third_model_ids=("solver", "verifier"))


async def test_worker_publishes_a_disjoint_signature_before_a_blocked_solver_route(monkeypatch):
    """A verifier-only route remains the working control while a solver-only route is blocked."""
    await _exercise_worker_publish_route(monkeypatch, third_model_ids=("verifier",))


async def test_later_verifier_publish_is_not_contaminated_by_a_blocked_solver_tail(monkeypatch):
    """A shared route and its verifier successor progress while an earlier solver-only route is blocked."""
    args = make_args(
        rollout_batch_size=1,
        async_data_buffer_capacity_factor=1.0,
        megatron_config=encode_megatron_config("solver", "verifier"),
    )
    fn = make_fn(monkeypatch, args, FakeDataSource())
    buffer = data_buffer.DefaultMultiDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None)
    )
    solver_inner = buffer._inners["solver"]
    await solver_inner.put(data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver")))
    await put_group(buffer, make_multi_policy_group(1, "solver"))
    fn._output = buffer
    second = fn._submit_publish(
        data_buffer.DataBufferInput(prompt_group=make_group(2), group=make_multi_policy_group(2, "solver"))
    )
    shared = fn._submit_publish(
        data_buffer.DataBufferInput(
            prompt_group=make_group(3),
            group=make_multi_policy_group(3, "solver", "verifier"),
        )
    )
    verifier_only = fn._submit_publish(
        data_buffer.DataBufferInput(prompt_group=make_group(4), group=make_multi_policy_group(4, "verifier"))
    )
    try:
        verifier_entries = [
            await asyncio.wait_for(buffer.get(trainer_model_id="verifier"), timeout=0.1) for _ in range(2)
        ]
        assert [data_buffer.first_sample(entry.group).group_index for entry in verifier_entries] == [3, 4]

        solver_entries = [await asyncio.wait_for(solver_inner.get(), timeout=0.1) for _ in range(4)]
        assert [data_buffer.first_sample(entry.group).group_index for entry in solver_entries] == [0, 1, 2, 3]
        await asyncio.wait_for(asyncio.gather(second, shared, verifier_only), timeout=0.1)
    finally:
        for task in (second, shared, verifier_only):
            if not task.done():
                task.cancel()
        await asyncio.gather(second, shared, verifier_only, return_exceptions=True)
        await buffer.aclose()


async def test_sample_backfill_does_not_exceed_the_shared_publish_budget(monkeypatch):
    """Completed sample callbacks cannot admit new generation while one publish owns the only group slot."""
    publish_started = asyncio.Event()
    publish_release = asyncio.Event()
    second_publish_started = asyncio.Event()
    close_release = asyncio.Event()
    publish_count = 0

    class _BlockingPublishBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            nonlocal publish_count
            publish_count += 1
            if publish_count == 1:
                publish_started.set()
                await publish_release.wait()
            else:
                second_publish_started.set()
                await close_release.wait()

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

        async def aclose(self) -> None:
            publish_release.set()
            close_release.set()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        for _ in group:
            sample_done_callback()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=generate)
    fn._output = _BlockingPublishBuffer()
    fn._worker = asyncio.create_task(fn._worker_loop())
    try:
        await asyncio.wait_for(publish_started.wait(), timeout=0.1)
        await asyncio.sleep(0.01)
        assert data_source.num_get_calls == 1
        publish_release.set()
        await asyncio.wait_for(second_publish_started.wait(), timeout=0.1)
        await asyncio.sleep(0.01)
        assert data_source.num_get_calls == 2
    finally:
        await fn.aclose()


async def test_worker_backpressures_policy_tickets_when_concurrency_exceeds_the_batch(monkeypatch):
    """A larger sample-concurrency budget waits on policy tickets instead of overflowing the dispatch lane."""
    releases = {group_index: asyncio.Event() for group_index in range(1, 5)}
    publish_started = {group_index: asyncio.Event() for group_index in range(1, 5)}
    close_release = asyncio.Event()
    args = make_args(
        rollout_batch_size=1,
        async_max_concurrent_samples=8,
        async_data_buffer_capacity_factor=1.0,
        megatron_config=encode_megatron_config("solver", "verifier"),
    )

    class _ObservedMultiBuffer(data_buffer.DefaultMultiDataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            publish_started[data_buffer.first_sample(input.group).group_index].set()
            await super().put(input)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = data_buffer.first_sample(group).group_index
        if group_index > 4:
            await close_release.wait()
            return make_multi_policy_group(group_index, "solver")
        await releases[group_index].wait()
        for _ in group:
            sample_done_callback()
        return make_multi_policy_group(group_index, "solver")

    fn = make_fn(
        monkeypatch,
        args,
        FakeDataSource(scripted=[make_group(group_index) for group_index in range(1, 5)]),
        generate=generate,
    )
    buffer = _ObservedMultiBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None)
    )
    await buffer._inners["solver"].put(
        data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
    )
    fn._output = buffer
    fn._worker = asyncio.create_task(fn._worker_loop())
    try:
        for group_index in range(1, 5):
            releases[group_index].set()
            await asyncio.wait_for(publish_started[group_index].wait(), timeout=0.1)
        await asyncio.sleep(0.01)
        assert not fn._worker.done()
    finally:
        close_release.set()
        for release in releases.values():
            release.set()
        await asyncio.gather(fn.aclose(), return_exceptions=True)


@pytest.mark.parametrize("group_budget", [1, 2])
async def test_sample_backfill_bounds_group_wrappers_during_rm_and_publish_stalls(monkeypatch, group_budget):
    """One replacement wave is bounded while completed samples wait for group RM and blocked publication."""
    initial_generation_started = asyncio.Event()
    replacement_started = asyncio.Event()
    rm_release = asyncio.Event()
    close_release = asyncio.Event()
    publish_started = [asyncio.Event() for _ in range(2 * group_budget)]
    publish_releases = [asyncio.Event() for _ in range(2 * group_budget)]
    publish_count = 0

    class _BlockingPublishBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            nonlocal publish_count
            index = publish_count
            publish_count += 1
            publish_started[index].set()
            await publish_releases[index].wait()

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

        async def aclose(self) -> None:
            rm_release.set()
            close_release.set()
            for release in publish_releases:
                release.set()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = data_buffer.first_sample(group).group_index
        if group_index <= 1000 + 2 * group_budget:
            for _ in group:
                sample_done_callback()
            if group_index == 1000 + 2 * group_budget:
                initial_generation_started.set()
            await rm_release.wait()
        else:
            replacement_started.set()
            await close_release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=group_budget), data_source, generate=generate)
    scheduler_wait_count = 0
    original_wait_for_progress = fn._scheduler.wait_for_progress

    async def record_wait_for_progress(
        pendings: set[asyncio.Task],
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        nonlocal scheduler_wait_count
        scheduler_wait_count += 1
        return await original_wait_for_progress(pendings)

    fn._scheduler.wait_for_progress = record_wait_for_progress
    fn._output = _BlockingPublishBuffer()
    fn._worker = asyncio.create_task(fn._worker_loop())
    try:
        await asyncio.wait_for(initial_generation_started.wait(), timeout=0.1)
        wait_count_at_capacity = scheduler_wait_count
        event_loop_progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(event_loop_progressed.set)
        await asyncio.sleep(0.01)
        assert data_source.num_get_calls == 2 * group_budget
        assert event_loop_progressed.is_set()
        assert scheduler_wait_count == wait_count_at_capacity

        rm_release.set()
        await asyncio.wait_for(publish_started[0].wait(), timeout=0.1)
        await asyncio.sleep(0.01)
        assert len(fn._active) + len(fn._publishing) <= 2 * group_budget
        assert len(fn._publishing) == 2 * group_budget
        assert data_source.num_get_calls == 2 * group_budget

        for index in range(group_budget):
            publish_releases[index].set()
            await asyncio.wait_for(publish_started[index + 1].wait(), timeout=0.1)
            await asyncio.sleep(0.01)
            assert data_source.num_get_calls == 2 * group_budget

        publish_releases[group_budget].set()
        await asyncio.wait_for(replacement_started.wait(), timeout=0.1)
        await asyncio.sleep(0.01)
        assert data_source.num_get_calls == 2 * group_budget + 1
    finally:
        await fn.aclose()


async def test_custom_buffer_owns_concurrent_publish_ordering(monkeypatch):
    """The worker forwards whole groups concurrently without imposing built-in policy routing on a custom buffer."""
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()

    class _RecordingBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            group_index = data_buffer.first_sample(input.group).group_index
            if group_index == 1:
                first_started.set()
                await first_release.wait()
            elif group_index == 2:
                second_started.set()

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

    def reject_policy_introspection(*args, **kwargs):
        raise AssertionError("custom buffer input must not be inspected by the rollout worker")

    monkeypatch.setattr(fully_async, "complete_trainer_model_ids", reject_policy_introspection, raising=False)
    fn = make_fn(monkeypatch, make_args(), FakeDataSource())
    fn._output = _RecordingBuffer()
    first = fn._submit_publish(data_buffer.DataBufferInput(prompt_group=make_group(1), group=make_group(1)))
    await asyncio.wait_for(first_started.wait(), timeout=0.1)
    second = fn._submit_publish(data_buffer.DataBufferInput(prompt_group=make_group(2), group=make_group(2)))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=0.1)
    finally:
        first_release.set()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_a_cancelled_publish_is_a_terminal_worker_failure(monkeypatch):
    """A buffer put that cancels itself fails the rollout instead of masquerading as owner teardown."""
    error = asyncio.CancelledError("custom output cancelled itself")

    class _CancelledPublishBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            raise error

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, rollout_submission_granularity="group"),
        FakeDataSource(),
    )
    fn._output = _CancelledPublishBuffer()
    fn._worker = asyncio.create_task(fn._worker_loop())
    try:
        with pytest.raises(RuntimeError, match="rollout publish.*cancelled") as failed:
            await fn._next_group(current_version=None, trainer_model_id=None)
        assert failed.value.__cause__ is error
    finally:
        with pytest.raises(RuntimeError, match="rollout publish.*cancelled") as closed:
            await fn.aclose()
        assert closed.value.__cause__ is error


async def test_aclose_preserves_a_publish_failure_ready_before_worker_resume(monkeypatch):
    """Teardown must surface a ready publish failure even before the worker collects its result."""
    error = RuntimeError("publish failed before worker resume")
    failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    publish_started = asyncio.Event()
    worker_saw_done = asyncio.Event()
    keep_worker_paused = asyncio.Event()
    real_wait = asyncio.wait

    class _FailingPublishBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            publish_started.set()
            await failed

        async def get(self, **context) -> data_buffer.DataBufferInput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, rollout_submission_granularity="group"),
        FakeDataSource(),
    )

    async def _pause_after_publish_finishes(tasks, **kwargs):
        done, pending = await real_wait(tasks, **kwargs)
        if any(task in fn._publishing and task.done() for task in done):
            worker_saw_done.set()
            await keep_worker_paused.wait()
        return done, pending

    monkeypatch.setattr(asyncio, "wait", _pause_after_publish_finishes)
    fn._output = _FailingPublishBuffer()
    fn._worker = asyncio.create_task(fn._worker_loop())
    await asyncio.wait_for(publish_started.wait(), timeout=0.1)
    failed.set_exception(error)
    await asyncio.wait_for(worker_saw_done.wait(), timeout=0.1)

    with pytest.raises(RuntimeError, match="publish failed before worker resume") as closed:
        await fn.aclose()
    assert closed.value is error
    assert fn._worker.done()
    assert not fn._active
    assert not fn._publishing


async def test_aclose_cancels_the_worker_and_its_active_generation(monkeypatch):
    """Rollout teardown owns both the producer worker and every generation task it submitted."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource(), generate=blocking_generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.wait_for(started.wait(), timeout=0.1)

    await fn.aclose()
    await fn.aclose()

    assert cancelled.is_set()
    assert fn._worker.done()
    await asyncio.gather(drain, return_exceptions=True)


async def test_dispose_closes_tasks_on_the_shared_rollout_loop(monkeypatch):
    """Synchronous actor teardown delegates cleanup to the loop that owns rollout tasks."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
    await asyncio.to_thread(call_rollout_function, fn, RolloutFnTrainInput(rollout_id=0))
    try:
        fn.dispose()
        assert fn._worker.done()
    finally:
        if not fn._worker.done():
            await asyncio.to_thread(run, fn.aclose())


async def test_aclose_finishes_after_its_first_waiter_is_cancelled(monkeypatch):
    """A cancelled teardown waiter must not orphan cleanup or make a later close return early."""
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_finished = asyncio.Event()

    class _SlowCloseBuffer(data_buffer.DataBuffer):
        async def put(self, input: data_buffer.DataBufferInput) -> None:
            raise AssertionError("unreachable")

        async def get(self, **context) -> data_buffer.DataBufferInput:
            raise AssertionError("unreachable")

        def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
            return {}

        async def aclose(self) -> None:
            close_started.set()
            await close_release.wait()
            close_finished.set()

    fn = make_fn(monkeypatch, make_args(), FakeDataSource())
    fn._output = _SlowCloseBuffer()
    first_close = asyncio.create_task(fn.aclose())
    await close_started.wait()
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    close_release.set()
    await fn.aclose()

    assert close_finished.is_set()


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
    assert len(submitted) > 1
    assert len(output.samples) == 1


def reject_group_1(args, group, **kwargs):
    keep = group[0].group_index != 1
    return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    args = make_args(
        rollout_batch_size=1,
        dynamic_sampling_filter_path=f"{__name__}.reject_group_1",
        async_unused_samples_handler="retry",
    )
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Dropped even with handler="retry": filter rejections bypass the unused handler.
    assert data_source.recycled == []
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_staleness_filter_off_before_the_first_weight_update(monkeypatch):
    """weight_version is None until the trainer pushes weights; staleness is unknown, not zero."""
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=0), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == []
    assert output.samples[0][0].group_index == 1
    assert "rollout/fully_async/max_staleness" not in output.metrics


# ── DataBuffer: staleness-bounded buffering ─────────────────────────


def make_buffer(max_groups=None, max_staleness=None):
    unused = []
    args = make_args(
        rollout_batch_size=1,  # capacity is factor * batch size; batch size 1 makes it count groups
        async_data_buffer_capacity_factor=max_groups or 1000.0,
        max_weight_staleness=max_staleness,
    )
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)
    )
    return buffer, unused


async def put_group(buffer, group):
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(prompt_group=group, group=group))


async def test_buffer_blocks_producer_when_full():
    buffer, _ = make_buffer(max_groups=2)
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    blocked = asyncio.create_task(put_group(buffer, make_group(3)))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    assert (await buffer.get()).group[0].group_index == 1
    await blocked
    assert (await buffer.get()).group[0].group_index == 2
    assert (await buffer.get()).group[0].group_index == 3


async def test_buffer_get_ignores_unknown_context_keys():
    """get(**context) lets the driver add keys without breaking existing buffers."""
    buffer, _ = make_buffer()
    await put_group(buffer, make_group(1))

    assert (await buffer.get(current_version=1, some_future_key=2)).group[0].group_index == 1


async def test_buffer_get_skips_groups_stale_at_consumption_time():
    """Both groups were fresh when buffered; only the version passed to get() decides."""
    buffer, unused = make_buffer(max_staleness=2)
    stale = make_group(1, weight_versions=["5"])
    await put_group(buffer, stale)
    await put_group(buffer, make_group(2, weight_versions=["9"]))

    assert (await buffer.get(current_version=10)).group[0].group_index == 2
    assert unused == [stale]
    assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1


async def test_buffer_staleness_metrics():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    assert "rollout/fully_async/buffer_avg_staleness" not in buffer.get_metrics()  # engine version never seen

    await put_group(buffer, make_group(2, weight_versions=["6"]))
    await put_group(buffer, make_group(3, weight_versions=["8"]))
    await buffer.get(current_version=10)  # pops group 1 and tracks the engine version clock
    metrics = buffer.get_metrics()
    assert metrics["rollout/fully_async/avg_staleness"] == 6.0  # consumed group 1: 10 - 4
    assert metrics["rollout/fully_async/buffer_avg_staleness"] == 3.0  # buffered groups 2, 3: (4 + 2) / 2
    assert metrics["rollout/fully_async/buffer_max_staleness"] == 4


def make_multi_policy_group(group_index: int, *trainer_model_ids: str) -> list[list[Sample]]:
    """One prompt group whose every trajectory produces samples for the named policy models."""
    return [
        [replace(sample, trainer_model_id=trainer_model_id) for trainer_model_id in trainer_model_ids]
        for sample in make_group(group_index)
    ]


def make_multi_buffer(
    *model_ids: str,
    max_staleness=None,
    paths_per_model=None,
    max_groups: int | None = None,
    rollout_batch_size: int = 1,
):
    unused = []
    args = make_args(
        rollout_batch_size=rollout_batch_size,
        async_data_buffer_capacity_factor=(1000.0 if max_groups is None else max_groups / rollout_batch_size),
        max_weight_staleness=max_staleness,
        megatron_config=encode_megatron_config(*model_ids),
        custom_async_data_buffer_path_per_model=paths_per_model,
    )
    buffer = data_buffer.DefaultMultiDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)
    )
    return buffer, unused


class TestPerPolicyQueues:
    async def test_a_full_policy_does_not_block_a_sibling_training_batch(self):
        """A bounded skew window lets another policy collect one complete batch before global backpressure."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1, rollout_batch_size=2)
        solver_inner = buffer._inners["solver"]
        await solver_inner.put(
            data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
        )
        await put_group(buffer, make_multi_policy_group(1, "solver"))

        async def _put_first_batch() -> None:
            await put_group(buffer, make_multi_policy_group(2, "solver", "verifier"))
            await put_group(buffer, make_multi_policy_group(3, "solver", "verifier"))

        first_batch = asyncio.create_task(_put_first_batch())
        third_put: asyncio.Task | None = None
        third_verifier: asyncio.Task | None = None
        try:
            verifier = [await asyncio.wait_for(buffer.get(trainer_model_id="verifier"), timeout=0.1) for _ in range(2)]
            await asyncio.wait_for(first_batch, timeout=0.1)
            assert [data_buffer.first_sample(entry.group).group_index for entry in verifier] == [2, 3]

            third_put = asyncio.create_task(put_group(buffer, make_multi_policy_group(4, "solver", "verifier")))
            third_verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
            await asyncio.sleep(0.01)
            assert not third_put.done()
            assert not third_verifier.done()

            prefilled = await solver_inner.get()
            assert data_buffer.first_sample(prefilled.group).group_index == 0
            await asyncio.sleep(0.01)
            assert not third_put.done()

            solver_only = await solver_inner.get()
            assert data_buffer.first_sample(solver_only.group).group_index == 1
            await asyncio.wait_for(third_put, timeout=0.1)
            verifier.append(await asyncio.wait_for(third_verifier, timeout=0.1))
            solver = [await asyncio.wait_for(solver_inner.get(), timeout=0.1) for _ in range(3)]

            assert [data_buffer.first_sample(entry.group).group_index for entry in verifier] == [2, 3, 4]
            assert [data_buffer.first_sample(entry.group).group_index for entry in solver] == [2, 3, 4]
        finally:
            for task in (first_batch, third_put, third_verifier):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first_batch, third_put, third_verifier) if task is not None),
                return_exceptions=True,
            )

    async def test_overlapping_policy_routes_have_independent_bounded_credits(self):
        """Distinct target signatures progress independently while their shared policy remains globally FIFO."""
        buffer, _ = make_multi_buffer("a", "b", "c", max_groups=1, rollout_batch_size=1)
        a_inner = buffer._inners["a"]
        await a_inner.put(data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "a")))

        await put_group(buffer, make_multi_policy_group(1, "a", "b"))
        first_b = await asyncio.wait_for(buffer.get(trainer_model_id="b"), timeout=0.1)
        await asyncio.wait_for(put_group(buffer, make_multi_policy_group(2, "a", "c")), timeout=0.1)
        first_c = await asyncio.wait_for(buffer.get(trainer_model_id="c"), timeout=0.1)
        blocked_ab = asyncio.create_task(put_group(buffer, make_multi_policy_group(3, "a", "b")))
        second_b = asyncio.create_task(buffer.get(trainer_model_id="b"))
        try:
            await asyncio.sleep(0.01)
            assert data_buffer.first_sample(first_b.group).group_index == 1
            assert data_buffer.first_sample(first_c.group).group_index == 2
            assert not blocked_ab.done()
            assert not second_b.done()
            a_metrics = buffer.get_metrics("a")
            assert a_metrics["rollout/fully_async/dispatch_pending"] == 2
            assert a_metrics["rollout/fully_async/dispatch_route_pending"] == 2
            for model_id in ("b", "c"):
                metrics = buffer.get_metrics(model_id)
                assert metrics["rollout/fully_async/dispatch_pending"] == 0
                assert metrics["rollout/fully_async/queue_size"] == 0
                assert metrics["rollout/fully_async/dispatch_route_pending"] == 2

            await a_inner.get()
            await asyncio.wait_for(blocked_ab, timeout=0.1)
            assert data_buffer.first_sample((await asyncio.wait_for(second_b, timeout=0.1)).group).group_index == 3
            a_entries = [await asyncio.wait_for(a_inner.get(), timeout=0.1) for _ in range(3)]

            assert [data_buffer.first_sample(entry.group).group_index for entry in a_entries] == [1, 2, 3]
        finally:
            for task in (blocked_ab, second_b):
                if not task.done():
                    task.cancel()
            await asyncio.gather(blocked_ab, second_b, return_exceptions=True)

    async def test_a_full_policy_does_not_block_a_complete_sibling(self):
        """A policy nobody drains must not keep another policy's completed group out of its queue."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1)
        await put_group(buffer, make_multi_policy_group(1, "solver"))
        waiting_verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        publishing = asyncio.create_task(put_group(buffer, make_multi_policy_group(2, "solver", "verifier")))

        try:
            done, _ = await asyncio.wait({waiting_verifier}, timeout=0.1)
            assert waiting_verifier in done
            verifier = waiting_verifier.result()
            assert [sample.group_index for sample in data_buffer.iter_samples(verifier.group)] == [2, 2]
            await asyncio.wait_for(publishing, timeout=0.1)

            first_solver = await buffer.get(trainer_model_id="solver")
            second_solver = await buffer.get(trainer_model_id="solver")
            assert [
                [sample.group_index for sample in data_buffer.iter_samples(entry.group)]
                for entry in (first_solver, second_solver)
            ] == [[1, 1], [2, 2]]
            assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0
            assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 0
        finally:
            for task in (waiting_verifier, publishing):
                if not task.done():
                    task.cancel()
            await asyncio.gather(waiting_verifier, publishing, return_exceptions=True)

    async def test_an_incomplete_full_policy_sibling_does_not_delay_a_complete_policy(self):
        """A short subgroup is dropped independently even when that policy's queue is already full."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1)
        await put_group(buffer, make_multi_policy_group(1, "solver"))
        first_solver, first_verifier = make_group(2)
        second_verifier = make_group(2)[1]
        first_solver.trainer_model_id = "solver"
        first_verifier.trainer_model_id = second_verifier.trainer_model_id = "verifier"

        await asyncio.wait_for(put_group(buffer, [[first_solver, first_verifier], [second_verifier]]), timeout=0.1)

        verifier = await buffer.get(trainer_model_id="verifier")
        solver = await buffer.get(trainer_model_id="solver")
        assert [sample.group_index for sample in data_buffer.iter_samples(verifier.group)] == [2, 2]
        assert [sample.group_index for sample in data_buffer.iter_samples(solver.group)] == [1, 1]
        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0

    async def test_an_incomplete_policy_subgroup_is_not_enqueued(self):
        """A missing policy trajectory must not reach training as a short prompt group."""
        buffer, unused = make_multi_buffer("solver", "verifier")
        solver_first, solver_second = make_group(1)
        [verifier_first, _] = make_group(1)
        solver_first.trainer_model_id = solver_second.trainer_model_id = "solver"
        verifier_first.trainer_model_id = "verifier"

        await put_group(buffer, [[solver_first, verifier_first], [solver_second]])
        solver = await buffer.get(trainer_model_id="solver")
        verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        await asyncio.sleep(0.01)

        assert solver.group == [[solver_first], [solver_second]]
        assert not verifier.done()
        assert unused == []
        verifier.cancel()

    async def test_an_outer_group_with_the_wrong_number_of_trajectories_is_refused(self):
        """The composite must not reinterpret an already-short outer prompt group as complete."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="must carry 2 trajectories"):
            await put_group(buffer, make_multi_policy_group(1, "solver", "verifier")[:1])

    async def test_a_later_complete_policy_group_wakes_a_waiting_consumer(self):
        """Dropping one short group must not mix it into the next prompt or starve its policy."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        first_solver, first_verifier = make_group(1)
        first_solver.trainer_model_id = "solver"
        first_verifier.trainer_model_id = "verifier"
        await put_group(buffer, [[first_solver, first_verifier], [make_tagged_sample(1, "solver")]])
        waiting = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        await asyncio.sleep(0.01)

        second_solver = make_group(2)
        second_verifier = make_group(2)
        for sample in second_solver:
            sample.trainer_model_id = "solver"
        for sample in second_verifier:
            sample.trainer_model_id = "verifier"
        await put_group(buffer, [[second_solver[0], second_verifier[0]], [second_solver[1], second_verifier[1]]])

        verifier = await waiting
        assert verifier.group == [[second_verifier[0]], [second_verifier[1]]]
        assert {sample.group_index for sample in data_buffer.iter_samples(verifier.group)} == {2}

    async def test_a_group_of_two_policies_lands_in_a_queue_of_each(self):
        """One generate call feeds both policies, and a shared queue would hand them each other's samples."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 1
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_a_policy_only_ever_drains_its_own_samples(self):
        """Training a policy on another policy's responses is the failure this queue split exists to stop."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        entry = await buffer.get(trainer_model_id="verifier")

        assert [sample.trainer_model_id for sample in data_buffer.iter_samples(entry.group)] == [
            "verifier",
            "verifier",
        ]

    async def test_a_policy_waits_for_its_own_queue_instead_of_taking_from_another(self):
        """A policy that consumed a queue it does not own would starve the policy that does."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "solver"))

        waiting = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        await asyncio.sleep(0.01)

        assert not waiting.done()
        waiting.cancel()

    async def test_an_untagged_sample_is_refused_at_the_split(self):
        """Every sample of a multi policy run is stamped by the generate function, so an unstamped one is a bug."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="must stamp every sample"):
            await put_group(buffer, make_group(1))

    async def test_a_sample_of_an_unknown_policy_is_refused(self):
        """Its groups would queue up in a buffer no trainer ever drains, and the run would simply stall."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await put_group(buffer, make_multi_policy_group(1, "solver", "reviewer"))

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0

    async def test_an_incomplete_unknown_policy_refuses_the_whole_group(self):
        """A short unknown subgroup is still invalid, and no valid sibling may be admitted first."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        first_solver, second_solver = make_group(1)
        reviewer = make_group(1)[0]
        first_solver.trainer_model_id = second_solver.trainer_model_id = "solver"
        reviewer.trainer_model_id = "reviewer"

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await put_group(buffer, [[first_solver, reviewer], [second_solver]])

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0

    async def test_cancelling_a_put_waiting_for_capacity_commits_no_policy_partially(self):
        """Capacity is reserved for every target together, so cancellation cannot publish one sibling alone."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1, rollout_batch_size=1)
        solver_inner = buffer._inners["solver"]
        await solver_inner.put(
            data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
        )
        await asyncio.wait_for(put_group(buffer, make_multi_policy_group(1, "solver", "verifier")), timeout=0.1)
        first_verifier = await asyncio.wait_for(buffer.get(trainer_model_id="verifier"), timeout=0.1)
        assert data_buffer.first_sample(first_verifier.group).group_index == 1

        waiting_put = asyncio.create_task(put_group(buffer, make_multi_policy_group(2, "solver", "verifier")))
        waiting_verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        try:
            await asyncio.sleep(0.01)
            assert not waiting_put.done()
            assert not waiting_verifier.done()
            waiting_put.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting_put

            await solver_inner.get()
            first_solver = await asyncio.wait_for(solver_inner.get(), timeout=0.1)
            assert data_buffer.first_sample(first_solver.group).group_index == 1
            await asyncio.sleep(0.01)
            assert not waiting_verifier.done()
        finally:
            for task in (waiting_put, waiting_verifier):
                if not task.done():
                    task.cancel()
            await asyncio.gather(waiting_put, waiting_verifier, return_exceptions=True)

    async def test_lane_capacity_preserves_fifo_for_a_later_overlapping_route(self):
        """A shared policy cannot bypass an earlier group while another target lane waits for space."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1, rollout_batch_size=1)
        solver_inner = buffer._inners["solver"]
        await solver_inner.put(
            data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
        )
        await put_group(buffer, make_multi_policy_group(1, "solver"))
        fillers = [
            asyncio.create_task(put_group(buffer, make_multi_policy_group(index, "solver"))) for index in range(2, 7)
        ]
        earlier: asyncio.Task[None] | None = None
        later: asyncio.Task[None] | None = None
        verifier: asyncio.Task[data_buffer.DataBufferInput] | None = None
        try:
            for _ in range(100):
                if len(buffer._dispatch_queues["solver"]) == buffer._dispatch_queue_capacity:
                    break
                await asyncio.sleep(0)
            assert len(buffer._dispatch_queues["solver"]) == buffer._dispatch_queue_capacity
            earlier = asyncio.create_task(put_group(buffer, make_multi_policy_group(7, "solver", "verifier")))
            await asyncio.sleep(0.01)
            later = asyncio.create_task(put_group(buffer, make_multi_policy_group(8, "verifier")))
            verifier = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
            await asyncio.sleep(0.01)
            assert not earlier.done()
            assert not later.done()
            assert not verifier.done()

            for task in fillers:
                task.cancel()
            await asyncio.gather(*fillers, return_exceptions=True)
            first = await asyncio.wait_for(verifier, timeout=0.1)
            assert data_buffer.first_sample(first.group).group_index == 7
            second = await asyncio.wait_for(buffer.get(trainer_model_id="verifier"), timeout=0.1)
            assert data_buffer.first_sample(second.group).group_index == 8
            await asyncio.wait_for(earlier, timeout=0.1)
            await asyncio.wait_for(later, timeout=0.1)
        finally:
            for task in (*fillers, earlier, later, verifier):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *fillers,
                *(task for task in (earlier, later, verifier) if task is not None),
                return_exceptions=True,
            )

    async def test_lane_capacity_does_not_block_a_later_disjoint_route(self):
        """A group waiting for policy lane space leaves an unrelated policy independently dispatchable."""
        buffer, _ = make_multi_buffer("solver", "verifier", "reviewer", max_groups=1, rollout_batch_size=1)
        solver_inner = buffer._inners["solver"]
        await solver_inner.put(
            data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
        )
        await put_group(buffer, make_multi_policy_group(1, "solver"))
        fillers = [
            asyncio.create_task(put_group(buffer, make_multi_policy_group(index, "solver")))
            for index in range(2, buffer._dispatch_queue_capacity + 3)
        ]
        earlier: asyncio.Task[None] | None = None
        later: asyncio.Task[None] | None = None
        try:
            for _ in range(100):
                if len(buffer._dispatch_queues["solver"]) == buffer._dispatch_queue_capacity:
                    break
                await asyncio.sleep(0)
            assert len(buffer._dispatch_queues["solver"]) == buffer._dispatch_queue_capacity
            earlier = asyncio.create_task(put_group(buffer, make_multi_policy_group(20, "solver", "verifier")))
            await asyncio.sleep(0.01)
            later = asyncio.create_task(put_group(buffer, make_multi_policy_group(21, "reviewer")))

            await asyncio.wait_for(later, timeout=0.1)
            reviewer = await asyncio.wait_for(buffer.get(trainer_model_id="reviewer"), timeout=0.1)
            assert data_buffer.first_sample(reviewer.group).group_index == 21
            assert not earlier.done()
        finally:
            for task in (*fillers, earlier, later):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *fillers,
                *(task for task in (earlier, later) if task is not None),
                return_exceptions=True,
            )

    async def test_cancelling_a_waiting_route_does_not_poison_later_policy_order(self):
        """A cancelled pre-admission group is skipped while the next shared route remains ordered and complete."""
        buffer, _ = make_multi_buffer("solver", "verifier", max_groups=1, rollout_batch_size=1)
        solver_inner = buffer._inners["solver"]
        await solver_inner.put(
            data_buffer.DataBufferInput(prompt_group=[], group=make_multi_policy_group(0, "solver"))
        )
        await put_group(buffer, make_multi_policy_group(1, "solver"))
        cancelled = asyncio.create_task(put_group(buffer, make_multi_policy_group(2, "solver")))
        later = asyncio.create_task(put_group(buffer, make_multi_policy_group(3, "solver", "verifier")))
        try:
            await asyncio.sleep(0.01)
            assert not cancelled.done()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled

            verifier = await asyncio.wait_for(buffer.get(trainer_model_id="verifier"), timeout=0.1)
            assert data_buffer.first_sample(verifier.group).group_index == 3

            solver_entries = [await asyncio.wait_for(solver_inner.get(), timeout=0.1) for _ in range(3)]
            assert [data_buffer.first_sample(entry.group).group_index for entry in solver_entries] == [0, 1, 3]
            await asyncio.wait_for(later, timeout=0.1)
        finally:
            for task in (cancelled, later):
                if not task.done():
                    task.cancel()
            await asyncio.gather(cancelled, later, return_exceptions=True)

    async def test_a_custom_inner_failure_poisons_the_composite_without_replaying_siblings(self):
        """An asynchronous admission failure is terminal and already admitted siblings remain exactly once."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        solver_inner = buffer._inners["solver"]
        solver_admitted = asyncio.Event()
        error = RuntimeError("verifier admission failed")
        closed: list[str] = []

        class _SolverInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                await solver_inner.put(input)
                solver_admitted.set()

            async def get(self, **context) -> data_buffer.DataBufferInput:
                return await solver_inner.get(**context)

            def get_metrics(self) -> dict[str, float]:
                return solver_inner.get_metrics()

            async def aclose(self) -> None:
                closed.append("solver")

        class _FailingVerifierInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                await solver_admitted.wait()
                raise error

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def aclose(self) -> None:
                closed.append("verifier")

        buffer._inners["solver"] = _SolverInner()
        buffer._inners["verifier"] = _FailingVerifierInner()
        await asyncio.wait_for(put_group(buffer, make_multi_policy_group(1, "solver", "verifier")), timeout=0.1)

        with pytest.raises(RuntimeError, match="verifier admission failed") as failed:
            await asyncio.wait_for(buffer.wait_failed(), timeout=0.1)
        assert failed.value is error
        with pytest.raises(RuntimeError, match="verifier admission failed") as rejected:
            await put_group(buffer, make_multi_policy_group(2, "solver", "verifier"))
        assert rejected.value is error
        first_solver, first_verifier = make_group(3)
        first_solver.trainer_model_id = "solver"
        first_verifier.trainer_model_id = "verifier"
        with pytest.raises(RuntimeError, match="verifier admission failed") as short_rejected:
            await put_group(buffer, [[first_solver], [first_verifier]])
        assert short_rejected.value is error
        with pytest.raises(RuntimeError, match="verifier admission failed") as unreadable:
            await buffer.get(trainer_model_id="solver")
        assert unreadable.value is error

        solver = await solver_inner.get()
        no_duplicate = asyncio.create_task(solver_inner.get())
        try:
            assert data_buffer.first_sample(solver.group).group_index == 1
            await asyncio.sleep(0.01)
            assert not no_duplicate.done()
        finally:
            no_duplicate.cancel()
            await asyncio.gather(no_duplicate, return_exceptions=True)

        for _ in range(2):
            with pytest.raises(RuntimeError, match="verifier admission failed") as close_failed:
                await buffer.aclose()
            assert close_failed.value is error
        assert sorted(closed) == ["solver", "verifier"]

    async def test_aclose_preserves_an_inner_failure_ready_before_dispatcher_resume(self):
        """Composite teardown must surface a ready inner failure before cancelling its dispatcher."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        error = RuntimeError("inner failed before dispatcher resume")
        failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        put_started = asyncio.Event()

        class _FailingInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                put_started.set()
                await failed

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

        buffer._inners["verifier"] = _FailingInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))
        await asyncio.wait_for(put_started.wait(), timeout=0.1)
        failed_waiter = asyncio.create_task(buffer.wait_failed())
        close_task: asyncio.Task[None]
        async with buffer._condition:
            close_task = asyncio.create_task(buffer.aclose())
            for _ in range(100):
                if buffer._closing:
                    break
                await asyncio.sleep(0)
            assert buffer._closing
            failed.set_exception(error)
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="inner failed before dispatcher resume") as waited:
            await failed_waiter
        with pytest.raises(RuntimeError, match="inner failed before dispatcher resume") as closed:
            await close_task
        assert waited.value is error
        assert closed.value is error
        assert all(task.done() for task in buffer._dispatch_tasks.values())

    async def test_aclose_preserves_an_inner_cancellation_cleanup_failure(self):
        """A pending inner that fails during owner cancellation remains the terminal close cause."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        error = RuntimeError("inner cancellation cleanup failed")
        put_started = asyncio.Event()

        class _CleanupFailingInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                put_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise error from None

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

        buffer._inners["verifier"] = _CleanupFailingInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))
        await asyncio.wait_for(put_started.wait(), timeout=0.1)

        with pytest.raises(RuntimeError, match="inner cancellation cleanup failed") as closed:
            await buffer.aclose()
        assert closed.value is error
        assert all(task.done() for task in buffer._dispatch_tasks.values())

    async def test_a_custom_inner_background_failure_is_supervised(self):
        """Every custom inner failure channel is watched even when its puts succeed."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        error = RuntimeError("verifier background failed")
        closed = asyncio.Event()

        class _BackgroundFailingInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                return None

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def wait_failed(self) -> None:
                raise error

            async def aclose(self) -> None:
                closed.set()

        buffer._inners["verifier"] = _BackgroundFailingInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))

        with pytest.raises(RuntimeError, match="verifier background failed") as failed:
            await asyncio.wait_for(buffer.wait_failed(), timeout=0.1)
        assert failed.value is error
        with pytest.raises(RuntimeError, match="verifier background failed") as close_failed:
            await buffer.aclose()
        assert close_failed.value is error
        assert closed.is_set()

    async def test_aclose_preserves_a_ready_background_failure_before_watcher_resume(self):
        """Closing a composite cannot replace a ready custom failure with its generic close sentinel."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        error = RuntimeError("background failed before watcher resume")
        failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        watcher_started = asyncio.Event()

        class _FailingWatcherInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                return None

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def wait_failed(self) -> None:
                watcher_started.set()
                await failed

        buffer._inners["verifier"] = _FailingWatcherInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))
        await asyncio.wait_for(watcher_started.wait(), timeout=0.1)
        failed_waiter = asyncio.create_task(buffer.wait_failed())
        close_task: asyncio.Task[None]
        async with buffer._condition:
            close_task = asyncio.create_task(buffer.aclose())
            for _ in range(100):
                if buffer._closing:
                    break
                await asyncio.sleep(0)
            assert buffer._closing
            failed.set_exception(error)
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="background failed before watcher resume") as waited:
            await failed_waiter
        with pytest.raises(RuntimeError, match="background failed before watcher resume") as closed:
            await close_task
        assert waited.value is error
        assert closed.value is error
        assert all(task.done() for task in buffer._failure_tasks.values())

    async def test_aclose_rejects_a_ready_background_watcher_return(self, monkeypatch: pytest.MonkeyPatch):
        """A custom failure watcher that returned before outer resume remains a terminal contract error."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        returned: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        watcher_child_done = asyncio.Event()
        watcher_started = asyncio.Event()
        original_shield = asyncio.shield

        async def gated_shield(awaitable: asyncio.Future[None]) -> None:
            if isinstance(awaitable, asyncio.Task) and awaitable.get_coro().cr_code.co_name == "_wait_inner_failure":
                await original_shield(awaitable)
                watcher_child_done.set()
                await asyncio.Event().wait()
            else:
                await original_shield(awaitable)

        monkeypatch.setattr(asyncio, "shield", gated_shield)

        class _ReturningWatcherInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                return None

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def wait_failed(self) -> None:
                watcher_started.set()
                await returned

        buffer._inners["verifier"] = _ReturningWatcherInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))
        await asyncio.wait_for(watcher_started.wait(), timeout=0.1)
        returned.set_result(None)
        await asyncio.wait_for(watcher_child_done.wait(), timeout=0.1)
        failed_waiter = asyncio.create_task(buffer.wait_failed())
        close_task = asyncio.create_task(buffer.aclose())

        with pytest.raises(RuntimeError, match="failure watcher.*returned normally") as waited:
            await failed_waiter
        with pytest.raises(RuntimeError, match="failure watcher.*returned normally") as closed:
            await close_task
        assert waited.value is closed.value
        assert all(task.done() for task in buffer._failure_tasks.values())

    async def test_an_inner_cancelled_error_is_latched_as_a_terminal_failure(self):
        """Only composite teardown may cancel a dispatcher; an inner cancellation is a run failure."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        error = asyncio.CancelledError("custom inner cancelled itself")

        class _CancelledInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                raise error

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

        buffer._inners["verifier"] = _CancelledInner()
        await put_group(buffer, make_multi_policy_group(1, "verifier", "verifier"))

        with pytest.raises(RuntimeError, match="verifier.*cancelled") as failed:
            await asyncio.wait_for(buffer.wait_failed(), timeout=0.1)
        assert failed.value.__cause__ is error

    async def test_closing_blocked_dispatchers_cancels_and_drains_them_once(self):
        """Repeated close settles every fixed dispatcher and each inner lifecycle exactly once."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        started = {model_id: asyncio.Event() for model_id in ("solver", "verifier")}
        cancelled: list[str] = []
        closed: list[str] = []

        class _BlockingInner(data_buffer.DataBuffer):
            def __init__(self, model_id: str):
                self._model_id = model_id

            async def put(self, input: data_buffer.DataBufferInput) -> None:
                started[self._model_id].set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append(self._model_id)
                    raise

            async def get(self, **context) -> data_buffer.DataBufferInput:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def aclose(self) -> None:
                closed.append(self._model_id)

        for model_id in started:
            buffer._inners[model_id] = _BlockingInner(model_id)

        await asyncio.wait_for(put_group(buffer, make_multi_policy_group(1, "solver", "verifier")), timeout=0.1)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=0.1)
        await buffer.aclose()
        await buffer.aclose()

        assert sorted(cancelled) == ["solver", "verifier"]
        assert sorted(closed) == ["solver", "verifier"]
        with pytest.raises(RuntimeError, match="closed"):
            await put_group(buffer, make_multi_policy_group(2, "solver", "verifier"))

    async def test_close_finishes_after_its_first_waiter_is_cancelled(self):
        """Composite close stays owned after caller cancellation and a later waiter observes completion."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        close_started = {model_id: asyncio.Event() for model_id in ("solver", "verifier")}
        close_release = asyncio.Event()
        closed: list[str] = []

        class _SlowCloseInner(data_buffer.DataBuffer):
            def __init__(self, model_id: str):
                self._model_id = model_id

            async def put(self, input: data_buffer.DataBufferInput) -> None:
                raise AssertionError("unreachable")

            async def get(self, **context) -> data_buffer.DataBufferInput:
                raise AssertionError("unreachable")

            def get_metrics(self) -> dict[str, float]:
                return {}

            async def aclose(self) -> None:
                close_started[self._model_id].set()
                await close_release.wait()
                closed.append(self._model_id)

        for model_id in close_started:
            buffer._inners[model_id] = _SlowCloseInner(model_id)

        first_close = asyncio.create_task(buffer.aclose())
        await asyncio.gather(*(event.wait() for event in close_started.values()))
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        close_release.set()
        await buffer.aclose()

        assert sorted(closed) == ["solver", "verifier"]

    async def test_the_prompt_group_of_a_split_group_stays_whole(self):
        """Recycling a rejected group resubmits prompts, which are not owned by either policy."""
        buffer, unused = make_multi_buffer("solver", "verifier", max_staleness=0)
        group = make_multi_policy_group(1, "solver", "verifier")
        for sample in data_buffer.iter_samples(group):
            sample.weight_versions = [
                WeightVersionsPerCall(spans=[WeightVersionSpan(version="1", abs_start=0, abs_end=1)])
            ]

        await put_group(buffer, group)
        drained = asyncio.create_task(buffer.get(current_version=9, trainer_model_id="solver"))
        await asyncio.sleep(0.01)

        assert unused == [group]
        drained.cancel()

    async def test_getting_for_a_policy_this_run_does_not_train_is_refused(self):
        """A typo in the trainer's model id would wait forever on a queue that is never fed."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await buffer.get(trainer_model_id="reviewer")

    async def test_every_policy_of_the_config_gets_a_queue_of_its_own(self):
        """The queues are built from --megatron-config, so a policy missing one has nowhere to put its groups."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 0

    async def test_a_policy_reads_and_resets_only_its_own_metric_window(self):
        """Draining one policy used to read and clear every policy's counters, moving them onto the wrong curve."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        solver_metrics = buffer.get_metrics("solver")

        assert set(solver_metrics) == {key for key in solver_metrics if not key.startswith(("solver/", "verifier/"))}
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_an_inner_buffer_is_told_which_policy_asks_for_a_group(self):
        """A custom per policy buffer cannot filter or account by policy if the composite eats that context."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        seen: list[dict] = []

        class _RecordingInner(data_buffer.DataBuffer):
            async def put(self, input: data_buffer.DataBufferInput) -> None:
                raise AssertionError("unreachable")

            async def get(self, **context):
                seen.append(context)
                return "entry"

            def get_metrics(self) -> dict[str, float]:
                return {}

        buffer._inners["solver"] = _RecordingInner()

        assert await buffer.get(current_version=4, trainer_model_id="solver") == "entry"
        assert seen == [{"current_version": 4, "trainer_model_id": "solver"}]


def make_tagged_sample(index: int, trainer_model_id: str | None) -> Sample:
    sample = make_group(index)[0]
    sample.trainer_model_id = trainer_model_id
    return sample


def split(group: data_buffer.Group, *, prompt_group=None) -> dict:
    return data_buffer._split_by_trainer_model_id(
        data_buffer.DataBufferInput(prompt_group=prompt_group if prompt_group is not None else [], group=group)
    )


class TestSplitByTrainerModelId:
    def test_a_group_of_one_policy_lands_whole_in_that_policy_alone(self):
        """The common group is a group of one policy, and every sample of it must reach that policy."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, second])

        assert list(ans) == ["solver"]
        assert ans["solver"].group == [first, second]

    def test_a_mixed_group_becomes_one_input_per_policy(self):
        """Each policy trains on its own samples only, so the group has to be cut along the tags."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([solver, verifier])

        assert list(ans) == ["solver", "verifier"]
        assert ans["solver"].group == [solver]
        assert ans["verifier"].group == [verifier]

    def test_the_samples_of_a_policy_keep_the_order_they_arrived_in(self):
        """Order carries the trajectory, and a reordered group trains on a reshuffled conversation."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, make_tagged_sample(3, "verifier"), second])

        assert ans["solver"].group == [first, second]

    def test_a_sub_group_of_a_multi_sample_trajectory_is_filtered_per_policy(self):
        """A generate function may return several samples per trajectory, and they need not share a policy."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver, verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_a_sub_group_no_sample_of_which_survives_is_dropped(self):
        """An empty sub-group is a trajectory with no samples, which the consumers cannot make sense of."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver], [verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_every_trajectory_of_the_group_is_split_on_its_own(self):
        """One finished group carries several trajectories, and each of them may mix policies differently."""
        first, second, third = (
            make_tagged_sample(1, "solver"),
            make_tagged_sample(2, "verifier"),
            make_tagged_sample(3, "solver"),
        )

        ans = split([[first, second], [third]])

        assert ans["solver"].group == [[first], [third]]
        assert ans["verifier"].group == [[second]]

    def test_the_prompt_group_travels_whole_into_every_split(self):
        """A rejected group is recycled by resubmitting its prompts, which belong to no policy in particular."""
        prompt_group = make_group(7)

        ans = split([make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")], prompt_group=prompt_group)

        assert ans["solver"].prompt_group is prompt_group
        assert ans["verifier"].prompt_group is prompt_group

    def test_an_untagged_sample_is_refused_before_anything_is_routed(self):
        """Nothing downstream can guess where an unstamped sample belongs, so the split is where it must stop."""
        with pytest.raises(AssertionError, match="must stamp every sample"):
            split([make_tagged_sample(1, "solver"), make_tagged_sample(2, None)])

    def test_a_group_with_no_sample_left_reaches_no_policy(self):
        """A group every filter emptied belongs to nobody, and inventing a key for it would assert on None."""
        assert split([]) == {}


class TestFilterGroup:
    def test_it_keeps_only_the_samples_of_the_policy_asked_for(self):
        """This is what stops a policy from training on another policy's responses."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer._filter_group([solver, verifier], trainer_model_id="solver") == [solver]

    def test_it_keeps_a_sub_group_that_still_has_samples(self):
        """A trajectory whose samples are split across policies survives on both sides, one sample each."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer._filter_group([[solver, verifier]], trainer_model_id="solver") == [[solver]]

    def test_it_drops_a_sub_group_that_lost_every_sample(self):
        """An empty list left in place would be a trajectory that consumers must special-case forever."""
        assert data_buffer._filter_group([[make_tagged_sample(1, "verifier")]], trainer_model_id="solver") == []

    def test_it_leaves_the_group_it_was_given_untouched(self):
        """It runs once per policy over the same group, so a mutating filter would eat the later policies' samples."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")
        group = [solver, [verifier]]

        data_buffer._filter_group(group, trainer_model_id="solver")

        assert group == [solver, [verifier]]


class RecordingBuffer(data_buffer.DefaultDataBuffer):
    constructed_with = None

    def __init__(self, input):
        super().__init__(input)
        RecordingBuffer.constructed_with = input


class TestPerPolicyBufferClass:
    def test_every_policy_keeps_the_built_in_buffer_by_default(self):
        """The flag is opt-in, so a run that does not pass it must compose exactly what it composed before."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert [type(inner) for inner in buffer._inners.values()] == [
            data_buffer.DefaultDataBuffer,
            data_buffer.DefaultDataBuffer,
        ]

    def test_a_named_policy_gets_the_class_the_flag_names(self):
        """Two policies can need different dataflow, which is the whole point of one buffer per policy."""
        buffer, _ = make_multi_buffer("solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"])

        assert type(buffer._inners["solver"]) is RecordingBuffer
        assert type(buffer._inners["verifier"]) is data_buffer.DefaultDataBuffer

    def test_the_custom_class_is_built_with_the_same_constructor_input(self):
        """A custom buffer owns staleness and recycling, so it needs the handler the built-in one gets."""
        buffer, unused = make_multi_buffer(
            "solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"]
        )

        assert RecordingBuffer.constructed_with.unused_handler_fn == unused.append
        assert RecordingBuffer.constructed_with.args is buffer._inners["verifier"]._args

    def test_a_policy_this_run_does_not_train_is_refused(self):
        """A typo would silently leave the policy it meant to configure on the built-in buffer."""
        with pytest.raises(AssertionError, match="train no policy of this run"):
            make_multi_buffer("solver", "verifier", paths_per_model=[f"reviewer={__name__}.RecordingBuffer"])


class TestParseDataBufferPaths:
    def test_it_maps_every_model_id_to_its_class_path(self):
        """This is the mapping the composite buffer is built from."""
        assert data_buffer._parse_data_buffer_paths(["solver=pkg.A", "verifier=pkg.B"]) == {
            "solver": "pkg.A",
            "verifier": "pkg.B",
        }

    def test_an_unset_flag_is_an_empty_mapping(self):
        """Default is every policy on the built-in buffer, which is the empty mapping."""
        assert data_buffer._parse_data_buffer_paths(None) == {}

    @pytest.mark.parametrize("entry", ["solver", "=pkg.A", "solver=", "solver =  "])
    def test_a_malformed_entry_is_refused(self, entry):
        """Silently ignoring it would run the policy on a buffer the user did not ask for."""
        with pytest.raises(ValueError, match="expected MODEL_ID=PATH"):
            data_buffer._parse_data_buffer_paths([entry])

    def test_the_whitespace_around_an_entry_is_not_part_of_the_names(self):
        """A shell-quoted entry keeps its spaces, and an import path with them resolves to nothing."""
        assert data_buffer._parse_data_buffer_paths([" solver = pkg.A "]) == {"solver": "pkg.A"}

    def test_a_model_id_named_twice_is_refused(self):
        """One of the two class paths would win silently, and which one is not something to guess."""
        with pytest.raises(ValueError, match="Duplicate model id"):
            data_buffer._parse_data_buffer_paths(["solver=pkg.A", "solver=pkg.B"])


class TestDataBufferArgumentRegistration:
    def test_the_per_model_flag_is_declared_by_the_rollout_function_that_uses_it(self):
        """The framework asks the selected rollout function for its flags, so this hook must declare it."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        parsed = parser.parse_args(["--custom-async-data-buffer-path-per-model", "solver=pkg.A", "verifier=pkg.B"])

        assert parsed.custom_async_data_buffer_path_per_model == ["solver=pkg.A", "verifier=pkg.B"]

    def test_a_run_that_never_passes_the_flag_leaves_every_policy_on_the_built_in_buffer(self):
        """The default has to be None, which _parse_data_buffer_paths reads as the empty mapping."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        assert parser.parse_args([]).custom_async_data_buffer_path_per_model is None


async def test_custom_data_buffer_path_replaces_default(monkeypatch):
    path = f"{__name__}.RecordingBuffer"
    args = make_args(custom_async_data_buffer_path=path, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert type(fn._output) is RecordingBuffer
    assert RecordingBuffer.constructed_with.unused_handler_fn == fn._recycle
    assert len(output.samples) == 2


class MultiPolicyDataSource(FakeDataSource):
    """Stamps every sample of a group with one policy, alternating, as a multi policy generate function does."""

    def get_samples(self, num_samples):
        [group] = super().get_samples(num_samples)
        for sample in group:
            sample.trainer_model_id = "a" if self.num_get_calls % 2 == 1 else "b"
        return [group]


class RecordingMultiBuffer(data_buffer.DefaultMultiDataBuffer):
    get_calls: list[dict] = []

    async def get(self, **context):
        RecordingMultiBuffer.get_calls.append(context)
        return await super().get(**context)


class TestBufferSelection:
    async def test_a_single_policy_run_keeps_the_plain_buffer(self, monkeypatch):
        """Every existing run goes through this line, and a per-policy buffer would key it under a model id."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert type(fn._output) is data_buffer.DefaultDataBuffer

    async def test_a_multi_policy_run_defaults_to_the_per_policy_buffer(self, monkeypatch):
        """One shared queue would hand a policy the groups another policy generated."""
        args = make_args(rollout_batch_size=1, megatron_config=encode_megatron_config("a", "b"))
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        output = await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is data_buffer.DefaultMultiDataBuffer
        assert [sample.trainer_model_id for group in output.samples for sample in group] == ["a", "a"]

    async def test_a_custom_buffer_still_wins_in_a_multi_policy_run(self, monkeypatch):
        """--custom-async-data-buffer-path is how a run replaces the queue, whatever the default would have been."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is RecordingMultiBuffer

    async def test_the_consumer_asks_the_buffer_for_the_policy_that_called_it(self, monkeypatch):
        """The queue is keyed by policy, so a consumer that forgets to name itself drains whoever answers first."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=4, trainer_model_id="a"))

        assert RecordingMultiBuffer.get_calls == [dict(current_version=4, trainer_model_id="a")]


async def test_worker_defaults_to_sample_granularity(monkeypatch):
    """Unset --rollout-submission-granularity backfills before a completed sample's group returns."""
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    # Report every sample of the still-pending group as finished.
    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await asyncio.sleep(0.01)

    assert data_source.num_get_calls == 2

    release.set()
    output = await drain
    assert len(output.samples) == 1


async def test_group_granularity_opts_the_worker_out_of_backfill(monkeypatch):
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1, rollout_submission_granularity="group")
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1
    # no callback wired at group level
    assert callbacks == [None]

    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 1


class TestRolloutFnContract:
    def test_it_is_a_rollout_fn_the_loader_accepts(self):
        """load_rollout_fn gates on issubclass(fn, BaseRolloutFn), so a class that forgets the
        base is rejected at startup no matter how complete its behaviour is."""
        assert issubclass(fully_async.FullyAsyncRolloutFn, BaseRolloutFn)

    def test_the_constructor_input_reaches_the_base(self, monkeypatch):
        """The base stores it as constructor_input; skipping super().__init__ leaves the
        attribute missing on every path that reads it."""
        data_source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

        assert fn.constructor_input.data_source is data_source


async def test_worker_bounds_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2
