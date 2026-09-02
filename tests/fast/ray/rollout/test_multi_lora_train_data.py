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
    rewards: list[float],
    start_index: int = 0,
):
    assert len(rewards) == n_samples
    group = []
    for k in range(n_samples):
        sample = make_sample(index=start_index + k, reward=rewards[k])
        sample.adapter = AdapterRef(name, slot)
        group.append(sample)
    return group


def make_batch():
    return [
        adapter_group("A", 0, 4, [1.0, 0.0, 1.0, 0.0], start_index=0),
        adapter_group("A", 0, 4, [1.0, 1.0, 1.0, 1.0], start_index=4),
        adapter_group("B", 1, 2, [3.0, 1.0], start_index=8),
    ]


def test_postprocess_extracts_batch_metadata_and_exact_batch_size():
    args = multi_lora_args()
    data, metadata = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 2})
    assert metadata["prompt_group_sizes"] == [4, 4, 2]
    assert metadata["dynamic_global_batch_size"] == 10  # exact batch size, no trim
    assert len(data) == 10  # flattened


def test_multi_lora_rejects_dp_indivisible_batch():
    args = multi_lora_args()
    with pytest.raises(ValueError, match="not divisible by dp_size"):
        postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 4})


def test_adapter_batch_without_tinker_lease_is_rejected():
    args = multi_lora_args()
    data, metadata = postprocess_rollout_data(args, make_batch(), train_parallel_config={"dp_size": 2})
    with pytest.raises(ValueError, match="batch lease"):
        convert_samples_to_train_data(
            args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
