"""--async-max-consecutive-aborted-groups: a fully async run fails after N aborted groups in a row."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["fully-async"])

import asyncio

import pytest

from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, train_input

from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class FakeEngines:
    """Weight version the engines serve; train_async publishes a step's weights before its drain starts."""

    def __init__(self):
        self.version = 0
        self.generating = 0
        self.started_at: dict[int, int] = {}  # group_index -> engine version at its first token


def make_generate(engines: FakeEngines, delay_of=lambda group_index: 0.0, abort_once=()):
    aborted = set()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        engines.started_at[group_index] = engines.version
        engines.generating += 1
        try:
            await asyncio.sleep(delay_of(group_index))
        finally:
            engines.generating -= 1
        for sample in group:
            sample.weight_versions = [
                WeightVersionsPerCall(
                    spans=[WeightVersionSpan(version=str(engines.started_at[group_index]), abs_start=0, abs_end=1)]
                )
            ]
            if group_index in abort_once and group_index not in aborted:
                sample.status = Sample.Status.ABORTED
            else:
                sample.status = Sample.Status.COMPLETED
            if sample_done_callback is not None:
                sample_done_callback()
        if group_index in abort_once:
            aborted.add(group_index)
        return group

    return generate


async def test_consecutive_aborted_groups_fail_the_run(monkeypatch):
    engines, data_source = FakeEngines(), FakeDataSource()
    generate = make_generate(engines, abort_once=set(range(1001, 1100)))
    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=2,
            max_weight_staleness=0,
            async_max_consecutive_aborted_groups=3,
            async_unused_samples_handler="drop",
        ),
        data_source,
        generate=generate,
    )
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await asyncio.wait_for(fn(train_input(rollout_id=0, weight_version=0)), timeout=10)
