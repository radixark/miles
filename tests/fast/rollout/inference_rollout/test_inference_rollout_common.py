from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
from miles.utils.types import Sample


class FakeGenerateState(inference_rollout_common.GenerateState):
    def __init__(self) -> None:
        self.args = Namespace(group_rm=False)
        self.aborted = False


async def test_aborted_group_releases_every_sample_completion_credit() -> None:
    state = FakeGenerateState()
    state.aborted = True
    group = [Sample(index=0), Sample(index=1)]
    completed_samples = 0

    def on_sample_done() -> None:
        nonlocal completed_samples
        completed_samples += 1

    result = await inference_rollout_common.generate_and_rm_group(
        state,
        group,
        sampling_params={},
        sample_done_callback=on_sample_done,
    )

    assert result == group
    assert completed_samples == len(group)


async def test_group_failure_cancels_and_settles_sibling_generation(monkeypatch) -> None:
    failure = RuntimeError("first parent failed")
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        if sample.index == 0:
            await sibling_started.wait()
            raise failure
        sibling_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            await release_sibling.wait()
            sibling_finished.set()
            raise

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await sibling_started.wait()
    await sibling_cancelled.wait()

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(group_task), timeout=0.01)
    finally:
        release_sibling.set()
        await sibling_finished.wait()

    with pytest.raises(RuntimeError) as error:
        await group_task

    assert error.value is failure
    assert sibling_finished.is_set()


async def test_group_cancellation_waits_for_child_generation_to_finish(monkeypatch) -> None:
    children_started = 0
    children_finished = 0
    all_children_started = asyncio.Event()
    release_children = asyncio.Event()
    all_children_finished = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        nonlocal children_finished, children_started
        children_started += 1
        if children_started == 2:
            all_children_started.set()
        try:
            await release_children.wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            await release_children.wait()
        children_finished += 1
        if children_finished == 2:
            all_children_finished.set()
        return sample

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await all_children_started.wait()
    group_task.cancel()

    try:
        await asyncio.sleep(0)
        assert not group_task.done()
        assert not child_cancelled.is_set()
    finally:
        release_children.set()
        await all_children_finished.wait()
        if not group_task.done():
            group_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await group_task

    with pytest.raises(asyncio.CancelledError):
        await group_task
    assert children_finished == 2


async def test_group_cancellation_chains_later_child_failure_after_drain(monkeypatch) -> None:
    failure = RuntimeError("child failed after cancellation")
    children_started = 0
    all_children_started = asyncio.Event()
    release_children = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        nonlocal children_started
        children_started += 1
        if children_started == 2:
            all_children_started.set()
        await release_children.wait()
        if sample.index == 0:
            raise failure
        sibling_finished.set()
        return sample

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await all_children_started.wait()
    group_task.cancel()
    release_children.set()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await group_task

    assert cancellation.value.__cause__ is failure
    assert sibling_finished.is_set()


async def test_group_cancellation_cancels_sibling_after_child_failure(monkeypatch) -> None:
    failure = RuntimeError("child failed after cancellation")
    children_started = 0
    all_children_started = asyncio.Event()
    release_failure = asyncio.Event()
    release_sibling = asyncio.Event()
    release_sibling_cleanup = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        nonlocal children_started
        children_started += 1
        if children_started == 2:
            all_children_started.set()
        if sample.index == 0:
            await release_failure.wait()
            raise failure
        try:
            await release_sibling.wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            await release_sibling_cleanup.wait()
            sibling_finished.set()
            raise
        sibling_finished.set()
        return sample

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await all_children_started.wait()
    group_task.cancel()
    release_failure.set()

    try:
        await asyncio.wait_for(sibling_cancelled.wait(), timeout=0.01)
        assert not group_task.done()
    finally:
        release_sibling.set()
        release_sibling_cleanup.set()
        await asyncio.gather(group_task, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await group_task

    assert cancellation.value.__cause__ is failure
    assert sibling_finished.is_set()


async def test_group_cancellation_prefers_failure_over_cancelled_child(monkeypatch) -> None:
    failure = RuntimeError("sibling failed after child cancellation")
    children_started = 0
    all_children_started = asyncio.Event()
    release_cancelled_child = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        nonlocal children_started
        children_started += 1
        if children_started == 2:
            all_children_started.set()
        if sample.index == 0:
            await release_cancelled_child.wait()
            child_cancelled.set()
            raise asyncio.CancelledError("child cancelled")
        await child_cancelled.wait()
        raise failure

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await all_children_started.wait()
    group_task.cancel()
    release_cancelled_child.set()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await group_task

    assert cancellation.value.__cause__ is failure


async def test_group_cancellation_cancels_sibling_after_child_cancellation(monkeypatch) -> None:
    child_cancellation = asyncio.CancelledError("child cancelled")
    children_started = 0
    all_children_started = asyncio.Event()
    release_cancelled_child = asyncio.Event()
    release_sibling = asyncio.Event()
    release_sibling_cleanup = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_and_rm(state, sample, sampling_params, evaluation=False):
        nonlocal children_started
        children_started += 1
        if children_started == 2:
            all_children_started.set()
        if sample.index == 0:
            await release_cancelled_child.wait()
            raise child_cancellation
        try:
            await release_sibling.wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            await release_sibling_cleanup.wait()
            sibling_finished.set()
            raise
        sibling_finished.set()
        return sample

    monkeypatch.setattr(inference_rollout_common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(inference_rollout_common, "policy_uses_routing_key", lambda args: False)
    group_task = asyncio.create_task(
        inference_rollout_common.generate_and_rm_group(
            FakeGenerateState(),
            [Sample(index=0), Sample(index=1)],
            sampling_params={},
        )
    )

    await all_children_started.wait()
    group_task.cancel()
    release_cancelled_child.set()

    try:
        await asyncio.wait_for(sibling_cancelled.wait(), timeout=0.01)
        assert not group_task.done()
    finally:
        release_sibling.set()
        release_sibling_cleanup.set()
        await asyncio.gather(group_task, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await group_task

    assert isinstance(cancellation.value.__cause__, asyncio.CancelledError)
    assert str(cancellation.value.__cause__) == "child cancelled"
    assert sibling_finished.is_set()
