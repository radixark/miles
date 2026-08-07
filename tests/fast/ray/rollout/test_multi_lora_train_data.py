"""Multi-LoRA train-data pipeline: BatchPlan-driven step metadata, no-trim
postprocessing, plan-authoritative slot routing, and per-group reward
normalization."""

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import AdapterRef


def multi_lora_args(**overrides):
    defaults = dict(
        multi_lora=True,
        grpo_std_normalization=True,
    )
    defaults.update(overrides)
    return make_args(**defaults)


def adapter_group(name: str, slot, n_samples: int, rewards: list[float], start_index: int = 0):
    assert len(rewards) == n_samples
    group = []
    for k in range(n_samples):
        sample = make_sample(index=start_index + k, reward=rewards[k])
        sample.adapter = AdapterRef(name=name, registration_id=f"reg-{name}", serving_version=1, slot=slot)
        group.append(sample)
    return group


def make_batch():
    """Two adapters, heterogeneous group sizes. B's samples are stamped with a
    STALE slot (None): under oversubscription the bind plan is authoritative."""
    return [
        adapter_group("A", 0, 4, [1.0, 0.0, 1.0, 0.0], start_index=0),
        adapter_group("A", 0, 4, [1.0, 1.0, 1.0, 1.0], start_index=4),
        adapter_group("B", None, 2, [3.0, 1.0], start_index=8),
    ]


def batch_plan_metadata():
    """What rollout_manager.generate derives from the wrapper's BatchPlan."""
    plan = [
        dict(name="A", registration_id="reg-A", bound_slot=0, evict=None, actual_rollout_count=8),
        dict(name="B", registration_id="reg-B", bound_slot=1, evict=None, actual_rollout_count=2),
    ]
    return {
        "step_slots": sorted(entry["bound_slot"] for entry in plan),
        "step_adapter_names": sorted(entry["name"] for entry in plan),
        "step_adapter_actual_counts": {entry["bound_slot"]: entry["actual_rollout_count"] for entry in plan},
        "adapter_name_by_slot": {entry["bound_slot"]: entry["name"] for entry in plan},
    }


def run_pipeline(dp_size: int = 2):
    args = multi_lora_args()
    data, metadata = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": dp_size})
    metadata.update(batch_plan_metadata())
    train_data = convert_samples_to_train_data(
        args,
        data,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    return data, metadata, train_data


def test_postprocess_extracts_group_sizes_and_never_trims():
    data, metadata, _ = run_pipeline()
    assert metadata["prompt_group_sizes"] == [4, 4, 2]
    # Batch shaping rides the rollout-side DP schedule: no dynamic-gbs
    # bookkeeping, and every selected sample survives postprocessing.
    assert "dynamic_global_batch_size" not in metadata
    assert len(data) == 10  # flattened, untrimmed


def test_dp_indivisible_batch_is_kept_whole():
    # 10 samples across dp_size=4: the rollout-side schedule distributes
    # micro-batches (not sample counts), so divisibility is no longer required.
    args = multi_lora_args()
    data, _ = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 4})
    assert len(data) == 10


def test_step_fields_come_from_the_batch_plan():
    _, _, train_data = run_pipeline()
    # B's samples were stamped slot=None (stale/unbound at generation time);
    # the bind plan routes them to slot 1 regardless.
    assert train_data["adapter_slots"] == [0] * 8 + [1] * 2
    assert train_data["step_slots"] == [0, 1]
    assert train_data["step_adapter_names"] == ["A", "B"]
    # ACTUAL rollout-execution counts drive step-time normalization (1/8, 1/2).
    assert train_data["step_adapter_actual_counts"] == {0: 8, 1: 2}
    assert train_data["adapter_name_by_slot"] == {0: "A", 1: "B"}
    assert train_data["prompt_group_sizes"] == [4, 4, 2]


def test_rewards_normalize_within_heterogeneous_groups():
    _, _, train_data = run_pipeline()
    rewards = train_data["rewards"]
    # Group boundaries: [0:4], [4:8], [8:10] — each zero-mean.
    for start, end in [(0, 4), (4, 8), (8, 10)]:
        assert sum(rewards[start:end]) == pytest.approx(0.0, abs=1e-6)
    # Constant group (all 1.0) normalizes to zeros, not NaN.
    assert rewards[4:8] == pytest.approx([0.0] * 4)
    # Singleton-free std normalization applied to group 1 (n=4, mixed).
    assert max(abs(r) for r in rewards[0:4]) > 0.5


def test_adapter_missing_from_bind_plan_is_an_error_not_a_stale_fallback():
    # A name absent from the plan means the batch and its selection disagree;
    # training on the stamped slot could write into another tenant's adapter.
    args = multi_lora_args()
    data, metadata = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 2})
    metadata.update(batch_plan_metadata())
    del metadata["adapter_name_by_slot"][1]  # drop B's mapping
    with pytest.raises(ValueError, match="no bind-plan slot"):
        convert_samples_to_train_data(
            args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
