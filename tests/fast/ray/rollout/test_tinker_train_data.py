"""Tinker conversion plane: BatchPlan → metadata (homogeneity enforced),
sample → train_data with authoritative slot routing and client channels, and
sample-level zero-weight DP padding that never enters the result plane."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import batch_plan_to_metadata, convert_samples_to_train_data
from miles.utils.types import AdapterRef, Sample


def plan_entry(name="A", slot=0, kind="forward_backward", op_id="op-A", loss=None):
    return dict(
        name=name,
        registration_id=f"r-{name}",
        bound_slot=slot,
        operation_id=op_id,
        operation_kind=kind,
        loss_spec=loss,
        sample_count=1,
    )


class TestBatchPlanToMetadata:
    def test_forward_backward_plan(self):
        metadata = batch_plan_to_metadata(
            [plan_entry("A", 0, loss={"loss_fn": "ppo"}), plan_entry("B", 3, op_id="op-B")]
        )
        assert metadata["batch_kind"] == "tinker"
        assert metadata["adapter_name_by_slot"] == {0: "A", 3: "B"}
        assert metadata["tinker_loss_by_slot"] == {0: {"loss_fn": "ppo"}, 3: {}}
        assert metadata["operation_by_slot"] == {0: "op-A", 3: "op-B"}
        assert "tinker_forward_only" not in metadata

    def test_all_forward_sets_the_flag(self):
        metadata = batch_plan_to_metadata([plan_entry(kind="forward")])
        assert metadata["tinker_forward_only"] is True

    def test_mixed_kinds_are_structurally_rejected(self):
        with pytest.raises(ValueError, match="homogeneous"):
            batch_plan_to_metadata([plan_entry("A", 0), plan_entry("B", 1, kind="forward")])
        with pytest.raises(ValueError, match="homogeneous"):
            batch_plan_to_metadata([plan_entry(kind="optim_step")])


def make_sample(name="A", index=0, stale_slot=9, loss_weights=None, advantages=None):
    sample = Sample(
        tokens=[1, 2, 3, 4],
        response_length=2,
        loss_mask=[1, 1],
        index=index,
        status=Sample.Status.COMPLETED,
        loss_weights=loss_weights,
        advantages=advantages,
    )
    sample.adapter = AdapterRef(name=name, registration_id=f"r-{name}", serving_version=1, slot=stale_slot)
    return sample


def convert(samples, metadata):
    args = SimpleNamespace(use_dynamic_global_batch_size=False)
    return convert_samples_to_train_data(
        args,
        samples,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestConvert:
    def test_tinker_batch_skips_rewards_and_routes_by_plan_slot(self):
        metadata = batch_plan_to_metadata([plan_entry("A", 5)])
        samples = [make_sample("A", i, stale_slot=9, loss_weights=[0.5, 1.5]) for i in range(2)]
        data = convert(samples, metadata)
        assert data["rewards"] == [0.0, 0.0]
        assert data["adapter_slots"] == [5, 5]  # the plan wins over the stale stamp
        assert data["loss_weights"] == [[0.5, 1.5], [0.5, 1.5]]
        assert data["sample_indices"] == [0, 1]
        assert data["batch_kind"] == "tinker"
        assert data["tinker_loss_by_slot"] == {5: {}}
        assert data["operation_by_slot"] == {5: "op-A"}
        assert "step_slots" not in data  # tinker never steps in-batch

    def test_unplanned_adapter_fails_loudly(self):
        metadata = batch_plan_to_metadata([plan_entry("A", 5)])
        with pytest.raises(ValueError, match="no BatchPlan slot"):
            convert([make_sample("ghost")], metadata)

    def test_mixed_channels_default_to_zeros(self):
        metadata = batch_plan_to_metadata([plan_entry("A", 0), plan_entry("B", 1, op_id="op-B")])
        samples = [
            make_sample("A", 0, loss_weights=[1.0, 1.0]),
            make_sample("B", 0, advantages=[0.5, -0.5]),
        ]
        data = convert(samples, metadata)
        assert data["loss_weights"] == [[1.0, 1.0], [0.0, 0.0]]
        assert data["advantages"] == [[0.0, 0.0], [0.5, -0.5]]

    def test_client_channels_survive_the_dp_shard_split(self):
        # The DP packager ships an explicit key list; a channel missing from it
        # silently reaches the loss as None ("needs per-token 'loss_weights'").
        from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw

        metadata = batch_plan_to_metadata([plan_entry("A", 0)])
        samples = [make_sample("A", i, loss_weights=[0.5, 1.5], advantages=[1.0, -1.0]) for i in range(2)]
        data = convert(samples, metadata)
        args = SimpleNamespace(balance_data=False, multi_lora_n_adapters=2)
        shards = split_train_data_by_dp_raw(args, data, dp_size=2)
        for shard in shards:
            assert shard["loss_weights"] == [[0.5, 1.5]]
            assert shard["advantages"] == [[1.0, -1.0]]


class TestPadding:
    """Sample-level zero-weight padding in ``postprocess_rollout_data``: tinker
    selections ride main's multi-LoRA dynamic-GBS branch, which requires the
    batch to be divisible by dp_size — pads make it so without trimming."""

    def tinker_args(self):
        return SimpleNamespace(
            multi_lora=True,
            use_dynamic_global_batch_size=True,
            disable_rollout_trim_samples=False,
            global_batch_size=8,
        )

    def samples(self, n):
        return [make_sample("A", i, loss_weights=[0.5, 1.5]) for i in range(n)]

    def postprocess(self, n, pad_to_dp=True, args=None):
        return postprocess_rollout_data(
            args or self.tinker_args(),
            self.samples(n),
            train_parallel_config={"dp_size": 4},
            pad_to_dp=pad_to_dp,
        )

    def test_pads_to_dp_size_with_inert_rows(self):
        data, _ = self.postprocess(n=2)
        assert len(data) == 4
        assert [s.index for s in data] == [0, 1, -1, -1]  # sentinel: filtered from the result plane
        assert data[2].loss_mask == [0, 0] and data[3].loss_weights == [0.0, 0.0]
        assert data[2].rollout_id is None
        assert data[0].loss_mask == [1, 1] and data[1].loss_weights == [0.5, 1.5]  # donors untouched
        assert all(s.adapter.name == "A" for s in data)  # pads clone the donor's routing

    def test_pads_to_the_next_multiple_not_just_dp_size(self):
        data, _ = self.postprocess(n=5)
        assert len(data) == 8
        assert [s.index for s in data] == [0, 1, 2, 3, 4, -1, -1, -1]

    def test_dynamic_gbs_matches_the_padded_length_and_nothing_is_trimmed(self):
        data, metadata = self.postprocess(n=2)
        assert metadata["dynamic_global_batch_size"] == 4 == len(data)

    def test_noop_when_batch_is_an_exact_multiple(self):
        data, metadata = self.postprocess(n=4)
        assert [s.index for s in data] == [0, 1, 2, 3]
        assert metadata["dynamic_global_batch_size"] == 4

    def test_non_tinker_path_keeps_default_trim_behavior(self):
        args = SimpleNamespace(
            multi_lora=False,
            use_dynamic_global_batch_size=False,
            disable_rollout_trim_samples=False,
            global_batch_size=2,
        )
        data, metadata = self.postprocess(n=5, pad_to_dp=False, args=args)
        assert [s.index for s in data] == [0, 1, 2, 3]  # trimmed, never padded
        assert "dynamic_global_batch_size" not in metadata
