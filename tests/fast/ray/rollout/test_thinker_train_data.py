"""Thinker batches through the train-data pipeline: BatchPlan-derived
metadata (step filtering, kind partitioning, loss specs), reward bypass, and
client channel packaging."""

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import (
    _package_shards,
    batch_plan_to_metadata,
    convert_samples_to_train_data,
)
from miles.utils.types import AdapterRef


def plan_entry(name, slot, kind="multi_lora_train", step=True, loss_spec=None, count=2):
    return dict(
        name=name,
        registration_id=f"reg-{name}",
        bound_slot=slot,
        evict=None,
        actual_sample_count=count,
        actual_rollout_count=count,
        prompt_group_sizes=[1] * count,
        operation_id=None if kind == "multi_lora_train" else f"op-{name}",
        operation_kind=kind,
        batch_id=None,
        step_after_backward=step,
        loss_spec=loss_spec,
    )


class TestBatchPlanMetadata:
    def test_accumulate_only_entries_never_step_but_keep_routing(self):
        plan = [
            plan_entry("A", 0, kind="forward_backward", step=False, loss_spec={"loss_fn": "cross_entropy"}),
            plan_entry("B", 1, kind="forward_backward", step=False, loss_spec={"loss_fn": "ppo"}),
        ]
        metadata = batch_plan_to_metadata(plan)
        assert metadata["step_slots"] == [] and metadata["step_adapter_names"] == []
        assert metadata["step_adapter_actual_counts"] == {}
        # Token routing still needs every selected adapter's slot.
        assert metadata["adapter_name_by_slot"] == {0: "A", 1: "B"}
        assert metadata["batch_kind"] == "thinker"
        assert metadata["adapter_loss_by_slot"] == {0: {"loss_fn": "cross_entropy"}, 1: {"loss_fn": "ppo"}}
        assert metadata["operation_by_slot"] == {0: "op-A", 1: "op-B"}

    def test_native_plan_keeps_the_fused_step_behavior(self):
        plan = [plan_entry("A", 0), plan_entry("B", 1)]
        metadata = batch_plan_to_metadata(plan)
        assert metadata["step_slots"] == [0, 1]
        assert metadata["step_adapter_actual_counts"] == {0: 2, 1: 2}
        assert "batch_kind" not in metadata and "adapter_loss_by_slot" not in metadata

    def test_mixed_kind_selection_is_a_bug(self):
        plan = [plan_entry("A", 0), plan_entry("B", 1, kind="forward_backward", step=False)]
        with pytest.raises(AssertionError, match="mixes input modes"):
            batch_plan_to_metadata(plan)


def thinker_sample(index, slot, *, loss_weights=None, advantages=None):
    sample = make_sample(index=index, reward=None)
    sample.adapter = AdapterRef(name="X", registration_id="reg-X", serving_version=1, slot=slot)
    sample.loss_weights = loss_weights
    sample.advantages = advantages
    return sample


def convert_thinker(samples, metadata):
    return convert_samples_to_train_data(
        make_args(multi_lora=True),
        samples,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestThinkerConversion:
    def test_rewards_bypass_and_channel_extraction(self):
        plan = [plan_entry("X", 0, kind="forward_backward", step=False, loss_spec={"loss_fn": "cross_entropy"})]
        metadata = batch_plan_to_metadata(plan)
        samples = [
            thinker_sample(0, 0, loss_weights=[0.5] * 4),
            thinker_sample(1, 0, advantages=[1.0] * 4),  # no weights: defaults to zeros
        ]
        train_data = convert_thinker(samples, metadata)

        # No reward post-processing ran: rewardless samples convert cleanly.
        assert train_data["rewards"] == [0.0, 0.0]
        assert train_data["loss_weights"] == [[0.5] * 4, [0.0] * 4]
        assert train_data["advantages"] == [[0.0] * 4, [1.0] * 4]
        assert train_data["batch_kind"] == "thinker"
        assert train_data["adapter_loss_by_slot"] == {0: {"loss_fn": "cross_entropy"}}
        assert train_data["step_adapter_actual_counts"] == {}

    def test_shards_carry_the_thinker_keys(self):
        plan = [plan_entry("X", 0, kind="forward_backward", step=False, loss_spec={"loss_fn": "ppo"})]
        metadata = batch_plan_to_metadata(plan)
        samples = [thinker_sample(i, 0, loss_weights=[1.0] * 4) for i in range(4)]
        train_data = convert_thinker(samples, metadata)
        train_data["total_lengths"] = [len(t) for t in train_data["tokens"]]

        shards = _package_shards(
            make_args(multi_lora=True, multi_lora_n_adapters=4), train_data, partitions=[[0, 2], [1, 3]]
        )
        for shard in shards:
            assert len(shard["loss_weights"]) == 2  # row-partitioned
            assert shard["batch_kind"] == "thinker"  # batch-level, replicated
            assert shard["adapter_loss_by_slot"] == {0: {"loss_fn": "ppo"}}


def test_gather_groups_logprob_rows_per_operation():
    from miles.backends.megatron_utils.multi_lora_utils.utils import _gather_thinker_logprobs

    rollout_data = {
        "thinker_logprob_collector": {
            (0, 1): [-0.2],
            (0, 0): [-0.1],
            (1, 0): [-0.9],
        },
        "operation_by_slot": {0: "op-A", 1: "op-B"},
    }
    result = _gather_thinker_logprobs(rollout_data)
    assert result == {"op-A": [[-0.1], [-0.2]], "op-B": [[-0.9]]}


class TestForwardOperations:
    def test_forward_and_forward_backward_share_a_selection(self):
        plan = [
            plan_entry("A", 0, kind="forward_backward", step=False, loss_spec={"loss_fn": "cross_entropy"}),
            plan_entry("B", 1, kind="forward", step=False),
        ]
        metadata = batch_plan_to_metadata(plan)
        assert metadata["batch_kind"] == "thinker"
        assert metadata["forward_only_slots"] == [1]
        assert metadata["step_slots"] == []

    def test_multi_lora_selection_never_mixes_with_thinker(self):
        plan = [plan_entry("A", 0), plan_entry("B", 1, kind="forward", step=False)]
        with pytest.raises(AssertionError, match="mixes input modes"):
            batch_plan_to_metadata(plan)
