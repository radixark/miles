from types import SimpleNamespace

import pytest

from miles.ray.multi_lora.residency import ResidentBinding
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.multi_lora.rollout_fn import batch_plan_to_metadata
from miles.utils.operation_contract import BatchExecutionLease
from miles.utils.types import AdapterRef, Sample


def plan_lease(batch_plan) -> BatchExecutionLease:
    return BatchExecutionLease(
        dispatch_id="lease-test",
        bindings_by_operation=tuple(
            (
                entry["operation_id"],
                ResidentBinding((entry["name"], entry["registration_id"]), entry["bound_slot"]),
            )
            for entry in batch_plan
        ),
    )


def plan_metadata(batch_plan) -> dict:
    return batch_plan_to_metadata(batch_plan, plan_lease(batch_plan))


def plan_entry(name="A", slot=0, kind="forward_backward", op_id="op-A", loss=None, sample_count=1):
    return dict(
        name=name,
        registration_id=f"r-{name}",
        bound_slot=slot,
        operation_id=op_id,
        operation_kind=kind,
        loss_spec=loss,
        sample_count=sample_count,
    )


class TestBatchPlanToMetadata:
    def test_forward_backward_plan(self):
        plan = [plan_entry("A", 0, loss={"loss_fn": "ppo"}), plan_entry("B", 3, op_id="op-B")]
        metadata = batch_plan_to_metadata(plan, plan_lease(plan))
        assert metadata["batch_kind"] == "tinker"
        assert metadata["tinker_operation_lanes"] == [0, 1]
        assert metadata["tinker_loss_by_lane"] == {0: {"loss_fn": "ppo"}, 1: {}}
        assert metadata["operation_by_lane"] == {0: "op-A", 1: "op-B"}
        assert metadata["registration_by_lane"] == {0: ("A", "r-A"), 1: ("B", "r-B")}
        assert metadata["batch_execution_lease"]["bindings_by_operation"] == [
            ["op-A", ["A", "r-A", 0]],
            ["op-B", ["B", "r-B", 3]],
        ]
        assert "tinker_forward_only" not in metadata

    def test_lanes_expand_per_sample_counts(self):
        plan = [plan_entry("A", 0, sample_count=2), plan_entry("B", 3, op_id="op-B", sample_count=3)]
        metadata = batch_plan_to_metadata(plan, plan_lease(plan))
        assert metadata["tinker_operation_lanes"] == [0, 0, 1, 1, 1]

    def test_all_forward_sets_the_flag(self):
        plan = [plan_entry(kind="forward")]
        metadata = batch_plan_to_metadata(plan, plan_lease(plan))
        assert metadata["tinker_forward_only"] is True

    def test_mixed_kinds_are_structurally_rejected(self):
        plan = [plan_entry("A", 0), plan_entry("B", 1, kind="forward")]
        with pytest.raises(ValueError, match="homogeneous"):
            batch_plan_to_metadata(plan, plan_lease(plan))
        plan = [plan_entry(kind="optim_step")]
        with pytest.raises(ValueError, match="homogeneous"):
            batch_plan_to_metadata(plan, plan_lease(plan))


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
        metadata = plan_metadata([plan_entry("A", 5, sample_count=2)])
        samples = [make_sample("A", i, stale_slot=9, loss_weights=[0.5, 1.5]) for i in range(2)]
        data = convert(samples, metadata)
        assert data["rewards"] == [0.0, 0.0]
        assert data["adapter_slots"] == [5, 5]
        assert data["loss_weights"] == [[0.5, 1.5], [0.5, 1.5]]
        assert data["sample_indices"] == [0, 1]
        assert data["batch_kind"] == "tinker"
        assert data["tinker_operation_lanes"] == [0, 0]
        assert data["tinker_loss_by_lane"] == {0: {}}
        assert data["operation_by_lane"] == {0: "op-A"}
        assert data["registration_by_lane"] == {0: ("A", "r-A")}
        assert data["batch_execution_lease"]["bindings_by_operation"] == [["op-A", ["A", "r-A", 5]]]
        assert "step_slots" not in data

    def test_two_operations_may_share_one_physical_slot(self):
        plan = [
            plan_entry("A", 5, op_id="op-A1"),
            plan_entry("A", 5, op_id="op-A2", loss={"loss_fn": "ppo"}),
        ]
        metadata = plan_metadata(plan)
        assert metadata["operation_by_lane"] == {0: "op-A1", 1: "op-A2"}
        assert metadata["tinker_loss_by_lane"] == {0: {}, 1: {"loss_fn": "ppo"}}
        samples = [make_sample("A", 0, loss_weights=[1.0, 1.0]), make_sample("A", 0, loss_weights=[2.0, 2.0])]
        data = convert(samples, metadata)
        assert data["adapter_slots"] == [5, 5]
        assert data["tinker_operation_lanes"] == [0, 1]

    def test_unplanned_adapter_fails_loudly(self):
        metadata = plan_metadata([plan_entry("A", 5)])
        with pytest.raises(ValueError, match="batch lease binds"):
            convert([make_sample("ghost")], metadata)

    def test_stale_same_name_registration_is_rejected_before_slot_routing(self):
        metadata = plan_metadata([plan_entry("A", 5)])
        stale = make_sample("A")
        stale.adapter = AdapterRef(name="A", registration_id="r-old", serving_version=1, slot=9)
        with pytest.raises(ValueError, match="registration"):
            convert([stale], metadata)

    def test_lease_binding_no_lane_references_is_a_plan_mismatch(self):
        metadata = plan_metadata([plan_entry("A", 5)])
        metadata["batch_execution_lease"]["bindings_by_operation"].append(["op-ghost", ["G", "r-G", 7]])
        with pytest.raises(ValueError, match="disagree"):
            convert([make_sample("A")], metadata)

    def test_mixed_channels_default_to_zeros(self):
        plan = [
            plan_entry("A", 0, loss={"loss_fn": "cross_entropy"}),
            plan_entry("B", 1, op_id="op-B", loss={"loss_fn": "importance_sampling"}),
        ]
        samples = [
            make_sample("A", 0, loss_weights=[1.0, 1.0]),
            make_sample("B", 0, advantages=[0.5, -0.5]),
        ]
        pure_ce_data = convert(samples[:1], plan_metadata(plan[:1]))
        assert "rollout_log_probs" not in pure_ce_data

        samples[1].rollout_log_probs = [-0.1, -0.2]
        data = convert(samples, plan_metadata(plan))
        assert data["loss_weights"] == [[1.0, 1.0], [0.0, 0.0]]
        assert data["advantages"] == [[0.0, 0.0], [0.5, -0.5]]
        assert data["rollout_log_probs"] == [[0.0, 0.0], [-0.1, -0.2]]

        reversed_data = convert(samples[::-1], plan_metadata(plan[::-1]))
        assert reversed_data["loss_weights"] == [[0.0, 0.0], [1.0, 1.0]]
        assert reversed_data["advantages"] == [[0.5, -0.5], [0.0, 0.0]]
        assert reversed_data["rollout_log_probs"] == [[-0.1, -0.2], [0.0, 0.0]]

    def test_legacy_batch_keeps_first_sample_optional_channel_semantics(self):
        samples = [make_sample("A"), make_sample("B")]
        for sample in samples:
            sample.adapter = None
        samples[1].rollout_log_probs = [-0.1, -0.2]

        data = convert_samples_to_train_data(
            SimpleNamespace(
                advantage_estimator="grpo", rewards_normalization=False, use_dynamic_global_batch_size=False
            ),
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert "rollout_log_probs" not in data

    def test_client_channels_survive_the_dp_shard_split(self):
        from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw

        metadata = plan_metadata([plan_entry("A", 0, sample_count=2)])
        samples = [make_sample("A", i, loss_weights=[0.5, 1.5], advantages=[1.0, -1.0]) for i in range(2)]
        data = convert(samples, metadata)
        args = SimpleNamespace(balance_data=False, multi_lora_n_adapters=2)
        shards = split_train_data_by_dp_raw(args, data, dp_size=2)
        for shard in shards:
            assert shard["loss_weights"] == [[0.5, 1.5]]
            assert shard["advantages"] == [[1.0, -1.0]]
            assert shard["tinker_operation_lanes"] == [0]
            assert shard["tinker_loss_by_lane"] == {0: {}}
            assert shard["operation_by_lane"] == {0: "op-A"}


class TestPadding:
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
        data, metadata = self.postprocess(n=2)
        assert metadata["dynamic_global_batch_size"] == len(data) == 4
        assert [s.index for s in data] == [0, 1, -1, -1]
        assert data[2].loss_mask == [0, 0] and data[3].loss_weights == [0.0, 0.0]
        assert data[2].rollout_id is None
        assert data[0].loss_mask == [1, 1] and data[1].loss_weights == [0.5, 1.5]
        assert all(s.adapter.name == "A" for s in data)

    def test_pads_to_the_next_multiple_not_just_dp_size(self):
        data, _ = self.postprocess(n=5)
        assert len(data) == 8
        assert [s.index for s in data] == [0, 1, 2, 3, 4, -1, -1, -1]

    def test_noop_when_batch_is_an_exact_multiple(self):
        data, metadata = self.postprocess(n=4)
        assert [s.index for s in data] == [0, 1, 2, 3]
        assert metadata["dynamic_global_batch_size"] == 4


class TestTinkerDispatchSummary:
    def test_summary_carries_operation_ids_and_lease(self):
        from miles.ray.rollout.train_data_conversion import tinker_dispatch_summary

        lease = {"dispatch_id": "d1", "bindings_by_operation": [["op-A", ["A", "r-A", 0]]]}
        train_data = {
            "batch_kind": "tinker",
            "operation_by_lane": {0: "op-A", 1: "op-B"},
            "batch_execution_lease": lease,
        }
        assert tinker_dispatch_summary(train_data) == {"operation_ids": ["op-A", "op-B"], "lease": lease}

    def test_non_tinker_batches_have_no_summary(self):
        from miles.ray.rollout.train_data_conversion import tinker_dispatch_summary

        assert tinker_dispatch_summary({"tokens": [[1]]}) is None
