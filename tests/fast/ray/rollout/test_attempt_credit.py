from copy import deepcopy

import ray.cloudpickle
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    _package_shards,
    convert_samples_to_train_data,
    process_rollout_data_shard,
)


def test_constraints_survive_conversion_serialization_and_reordered_dp_shards() -> None:
    args = make_args(rewards_normalization=False)
    samples = [make_sample(index=i) for i in range(3)]
    for sample in samples:
        sample.loss_mask = [1, 0, 1, 1]
    samples[1].metadata = {"non_positive_advantage_spans": [[1, 3]]}
    samples[2].metadata = {"non_positive_advantage_spans": [[0, 1], [3, 4]]}
    original_masks = [deepcopy(sample.loss_mask) for sample in samples]
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    assert data["non_positive_advantage_spans"] == [[], [[1, 3]], [[0, 1], [3, 4]]]
    assert data["loss_masks"] == original_masks
    assert ROLLOUT_DATA_VALUE_SPEC["non_positive_advantage_spans"].codec == "msgpack_ragged"

    data["total_lengths"] = [len(sample.tokens) for sample in samples]
    shards = _package_shards(args, data, [[2, 0], [1]])
    for shard in shards:
        partition = shard["partition"]
        # The Ray backend transports these Python values through cloudpickle.
        transported = ray.cloudpickle.loads(ray.cloudpickle.dumps(shard))
        received = process_rollout_data_shard(args, transported)
        assert received["sample_indices"] == partition
        assert received["non_positive_advantage_spans"] == [data["non_positive_advantage_spans"][i] for i in partition]
        assert received["loss_masks"] == [original_masks[i] for i in partition]


def test_unmarked_batches_do_not_gain_a_constraint_field() -> None:
    data = convert_samples_to_train_data(make_args(rewards_normalization=False), [make_sample()], {}, None, None)
    assert "non_positive_advantage_spans" not in data
