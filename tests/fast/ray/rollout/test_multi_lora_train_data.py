"""Multi-LoRA train-data pipeline: batch metadata extraction, exact dynamic
batch size, per-adapter batch loss scales, step stamping, and per-group reward
normalization with heterogeneous group sizes."""

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
        use_dynamic_global_batch_size=True,
        grpo_std_normalization=True,
    )
    defaults.update(overrides)
    return make_args(**defaults)


def adapter_group(
    name: str,
    slot: int,
    n_samples: int,
    adapter_global_batch_size: int,
    rewards: list[float],
    start_index: int = 0,
):
    assert len(rewards) == n_samples
    group = []
    for k in range(n_samples):
        sample = make_sample(index=start_index + k, reward=rewards[k])
        sample.adapter = AdapterRef(name, slot)
        sample.metadata = {"adapter_global_batch_size": adapter_global_batch_size}
        group.append(sample)
    return group


def make_batch():
    """Two adapters, heterogeneous group sizes: A steps this batch, B doesn't."""
    groups = [
        adapter_group("A", 0, 4, 16, [1.0, 0.0, 1.0, 0.0], start_index=0),
        adapter_group("A", 0, 4, 16, [1.0, 1.0, 1.0, 1.0], start_index=4),
        adapter_group("B", 1, 2, 32, [3.0, 1.0], start_index=8),
    ]
    groups[0][0].metadata["step_slots"] = [0]
    groups[0][0].metadata["step_adapter_names"] = ["A"]
    return groups


def run_pipeline(dp_size: int = 2):
    args = multi_lora_args()
    data, metadata = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": dp_size})
    train_data = convert_samples_to_train_data(
        args,
        data,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    return data, metadata, train_data


def test_postprocess_extracts_batch_metadata_and_exact_batch_size():
    data, metadata, _ = run_pipeline()
    assert metadata["prompt_group_sizes"] == [4, 4, 2]
    assert metadata["step_slots"] == [0]
    assert metadata["step_adapter_names"] == ["A"]
    assert metadata["dynamic_global_batch_size"] == 10  # exact batch size, no trim
    assert len(data) == 10  # flattened
    assert "step_slots" not in data[0].metadata  # lifted out


def test_multi_lora_rejects_dp_indivisible_batch():
    args = multi_lora_args()
    with pytest.raises(ValueError, match="not divisible by dp_size"):
        postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 4})


def test_step_fields():
    _, _, train_data = run_pipeline()
    assert train_data["adapter_slots"] == [0] * 8 + [1] * 2
    assert train_data["step_slots"] == [0]
    assert train_data["step_adapter_names"] == ["A"]
    # Only A steps: the trainer scales slot 0's accumulated gradient by 1/16.
    assert train_data["step_adapter_batch_sizes"] == {0: 16}
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
