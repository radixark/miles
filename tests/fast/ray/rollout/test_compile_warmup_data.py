from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.backends.torchtitan_utils.length_buckets import fixed_shape_length_bucket, fixed_shape_length_buckets
from miles.backends.training_utils.data.rollout import _num_microbatches_within_cap
from miles.ray.rollout.compile_warmup_data import compile_warmup_samples
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp


@pytest.mark.parametrize("balance_data", [False, True])
@pytest.mark.parametrize(
    ("seq_length", "max_tokens_per_gpu", "global_batch_size", "dp_size", "n_samples_per_prompt", "unreachable"),
    [
        # the TPU launcher's full and smoke recipes
        (8192, 7168, 128, 4, 8, set()),
        (8192, 7168, 32, 4, 4, set()),
        # a --seq-length off the power-of-two ladder is the top bucket itself
        (3000, 3000, 6, 2, 3, set()),
        # above the cap, only a lone sample longer than the cap gets a microbatch of that length
        (8192, 3000, 16, 2, 4, {8192}),
        # 1024 local samples of two tokens each already overshoot the smallest bucket
        (4096, 4096, 1024, 1, 8, {1024}),
    ],
)
def test_each_rank_packs_one_microbatch_of_the_requested_bucket(
    seq_length, max_tokens_per_gpu, global_batch_size, dp_size, n_samples_per_prompt, unreachable, balance_data
):
    """The trainer then compiles exactly the programs of each bucket, for one optimizer step's worth of samples."""
    args = make_args(
        global_batch_size=global_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        n_samples_per_prompt=n_samples_per_prompt,
        raw_torchtitan=SimpleNamespace(seq_length=seq_length),
        advantage_estimator="grpo",
        rewards_normalization=True,
        balance_data=balance_data,
    )

    samples_of_bucket = compile_warmup_samples(args, dp_size=dp_size)

    assert set(samples_of_bucket) == fixed_shape_length_buckets(seq_length) - unreachable
    for bucket, samples in samples_of_bucket.items():
        assert len(samples) == global_batch_size
        assert all(sample.response_length >= 1 for sample in samples)
        train_data = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        shards = split_train_data_by_dp(args, train_data, make_train_parallel_config(dp_size=dp_size))
        for shard in shards:
            lengths = [shard["total_lengths"][index] for index in shard["partition"]]
            assert len(lengths) == global_batch_size // dp_size
            assert _num_microbatches_within_cap(lengths, max_tokens_per_gpu) == 1
            assert fixed_shape_length_bucket(sum(lengths), seq_length) == bucket
