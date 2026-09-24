from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio

import pytest
from tests.fast.rollout.inference_rollout.test_sample_completion_backfill import Harness, make_args, make_group

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.filter_hub.base_types import MetricGatherer, aborted_exit_status, group_has_aborted
from miles.utils.types import Sample


def _aborted_group(group_index: int, exit_status: str | None) -> list[Sample]:
    group = make_group(group_index)
    group[0].status = Sample.Status.ABORTED
    group[0].reward = None
    if exit_status is not None:
        group[0].metadata["exit_status"] = exit_status
    return group


def test_group_has_aborted_handles_flat_and_nested_groups():
    assert not group_has_aborted(make_group(1))
    assert group_has_aborted(_aborted_group(1, "InfraError"))
    # v2 groups nest a list of leaves per prompt
    assert group_has_aborted([make_group(1), _aborted_group(2, None)])


def test_aborted_exit_status_reads_the_first_aborted_sample():
    assert aborted_exit_status(_aborted_group(1, "SandboxUnavailable")) == "SandboxUnavailable"
    assert aborted_exit_status(_aborted_group(1, None)) == "unknown"
    assert aborted_exit_status(make_group(1)) == "unknown"


def test_metric_gatherer_counts_aborted_drops_by_exit_status():
    gatherer = MetricGatherer()
    gatherer.on_aborted_group_drop(_aborted_group(1, "SandboxUnavailable"))
    gatherer.on_aborted_group_drop(_aborted_group(2, "SandboxUnavailable"))
    gatherer.on_aborted_group_drop(_aborted_group(3, None))
    gatherer.on_dynamic_filter_drop("zero_std")

    assert gatherer.collect() == {
        "rollout/aborted/drop_SandboxUnavailable": 2,
        "rollout/aborted/drop_unknown": 1,
        "rollout/dynamic_filter/drop_zero_std": 1,
    }


@pytest.mark.asyncio
async def test_sync_rollout_counts_an_aborted_drop_by_cause(monkeypatch):
    """The sync loop already drops an aborted group; it must also record why, or the cause never reaches a dashboard."""
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2, dynamic_sampling_filter_path=None))
    aborted = _aborted_group(999, "SandboxUnavailable")
    normal_data_source = harness.data_source

    def data_source(num_groups):
        # the first group the loop draws is the aborted one; every later draw is a normal group
        if not data_source.served_aborted:
            data_source.served_aborted = True
            return [aborted]
        return normal_data_source(num_groups)

    data_source.served_aborted = False
    task = asyncio.create_task(train.generate_rollout_async(harness.state, 0, data_source))

    # finish every submitted group as it appears until the loop has its batch
    while not task.done():
        for blocker in harness._blockers:
            blocker.set()
        await asyncio.sleep(0)

    output, _ = task.result()
    assert len(output.samples) == 2
    assert output.metrics["rollout/aborted/drop_SandboxUnavailable"] == 1
