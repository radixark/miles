from __future__ import annotations

import asyncio
from argparse import Namespace

import numpy as np
import pytest
from tests.fast.fixtures.object_store_fixtures import InMemoryStore

from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.stored_array import StoredArray
from miles.utils.types import Sample


class FakeGenerateState:
    def __init__(self, *, rollout_replay_via_object_store: bool):
        self.args = Namespace(
            rollout_replay_via_object_store=rollout_replay_via_object_store,
            sglang_router_policy="round_robin",
            sglang_enable_deterministic_inference=False,
            group_rm=False,
        )
        self.aborted = False


def make_group() -> list[Sample]:
    return [Sample(index=i, status=Sample.Status.PENDING) for i in range(2)]


@pytest.fixture
def slow_second_sample(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Sample 0 finishes at once; sample 1 finishes only when the returned event is set."""
    release_second = asyncio.Event()

    async def fake_generate_and_rm(state, sample, sampling_params, evaluation=False):
        if sample.index == 1:
            await release_second.wait()
        sample.rollout_routed_experts = np.full((3, 2, 2), sample.index, dtype=np.int32)
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(common, "generate_and_rm", fake_generate_and_rm)
    return release_second


async def test_a_finished_sample_is_stored_while_its_sibling_still_runs(
    in_memory_store: InMemoryStore, slow_second_sample: asyncio.Event
):
    """The group wait is where finished samples pile up, so storing at group completion would be too late."""
    group = make_group()
    task = asyncio.create_task(
        common.generate_and_rm_group(FakeGenerateState(rollout_replay_via_object_store=True), group, {})
    )
    while not isinstance(group[0].rollout_routed_experts, StoredArray):
        await asyncio.sleep(0.01)

    assert group[1].rollout_routed_experts is None
    slow_second_sample.set()
    finished = await task

    assert all(isinstance(sample.rollout_routed_experts, StoredArray) for sample in finished)
    np.testing.assert_array_equal(finished[1].rollout_routed_experts.handle.read(), np.full((3, 2, 2), 1))


@pytest.mark.parametrize(
    "rollout_replay_via_object_store, evaluation", [(False, False), (True, True)], ids=["flag-off", "eval"]
)
async def test_arrays_stay_inline_without_the_flag_or_for_eval(
    in_memory_store: InMemoryStore,
    slow_second_sample: asyncio.Event,
    rollout_replay_via_object_store: bool,
    evaluation: bool,
):
    slow_second_sample.set()

    finished = await common.generate_and_rm_group(
        FakeGenerateState(rollout_replay_via_object_store=rollout_replay_via_object_store),
        make_group(),
        {},
        evaluation=evaluation,
    )

    assert all(isinstance(sample.rollout_routed_experts, np.ndarray) for sample in finished)
    assert in_memory_store.objects == {}
