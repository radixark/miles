import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample


class RecordingSink:
    def __init__(self):
        self.events = []

    def attempt_start(self, sample):
        self.events.append(("attempt_start", sample.index))

    def gen_start(self, sample):
        self.events.append(("gen_start", sample.index))

    def attempt_end(self, sample):
        self.events.append(("attempt_end", sample.index))


@pytest.fixture
def lifecycle_sink():
    sink = RecordingSink()
    TrajectoryLifecycle().sink = sink
    yield sink
    TrajectoryLifecycle().sink = None


def make_state(generate_function):
    args = SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        sglang_router_policy="round_robin",
    )
    return SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
        generate_function=generate_function,
    )


async def test_attempt_ends_once_after_success(lifecycle_sink):
    async def successful_generate(input):
        return GenerateFnOutput(samples=input.sample)

    sample = Sample(index=1)
    result = await common.generate_and_rm(make_state(successful_generate), sample, sampling_params={})

    assert result is sample
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_once_after_abort(lifecycle_sink):
    async def unexpected_generate(_input):
        pytest.fail("aborted attempts must not generate")

    sample = Sample(index=1)
    state = make_state(unexpected_generate)
    state.aborted = True
    result = await common.generate_and_rm(state, sample, sampling_params={})

    assert result is sample
    assert sample.status == Sample.Status.ABORTED
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_when_generate_raises(lifecycle_sink):
    async def failing_generate(_input):
        raise RuntimeError("generate failed")

    sample = Sample(index=1)
    with pytest.raises(RuntimeError, match="generate failed"):
        await common.generate_and_rm(make_state(failing_generate), sample, sampling_params={})

    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_cancelled_group_sibling_ends_attempt(lifecycle_sink):
    all_started = asyncio.Event()
    never_finishes = asyncio.Event()
    started = 0
    cancelled = []

    async def generate(input):
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        try:
            await all_started.wait()
            if input.sample.index == 1:
                raise RuntimeError("sample failed")
            await never_finishes.wait()
        except asyncio.CancelledError:
            cancelled.append(input.sample.index)
            raise

    samples = [Sample(index=1), Sample(index=2)]
    with pytest.raises(RuntimeError, match="sample failed"):
        await common.generate_and_rm_group(make_state(generate), samples, sampling_params={})

    assert cancelled == [2]
    for sample in samples:
        assert [kind for kind, index in lifecycle_sink.events if index == sample.index] == [
            "attempt_start",
            "gen_start",
            "attempt_end",
        ]
