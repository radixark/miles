import math
from argparse import Namespace
from types import SimpleNamespace

import msgpack
import pytest
import ray.cloudpickle
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.utils.types import Sample


@pytest.mark.parametrize("codec", [msgpack, ray.cloudpickle])
def test_teacher_metadata_survives_sharding_serialization_and_batching(monkeypatch, codec):
    group = SimpleNamespace(size=1, rank=0)
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(cp=group, tp=group))
    args = Namespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        balance_data=False,
        use_dynamic_global_batch_size=False,
    )
    samples = [
        Sample(
            index=i,
            tokens=[i, i + 1],
            response_length=1,
            reward=0,
            train_metadata={"opd": {"ids": [[i, 0]], "logprobs": [[-0.2, -math.inf]]}},
        )
        for i in range(4)
    ]
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    for shard in split_train_data_by_dp_raw(args, data, dp_size=2):
        shard["partition"] = list(shard["partition"])
        shard = codec.loads(codec.dumps(shard))
        shard["tokens"] = [torch.tensor(tokens) for tokens in shard["tokens"]]
        shard["loss_masks"] = [torch.tensor(mask) for mask in shard["loss_masks"]]
        shard["max_seq_lens"] = [2, 2]
        shard["total_lengths"] = [2, 2]
        batch = get_batch(
            DataIterator(shard, micro_batch_indices=[[1, 0]]),
            ["tokens", "total_lengths", "response_lengths", "loss_masks", "max_seq_lens"],
            qkv_format="bshd",
        )
        for tokens, metadata in zip(batch["tokens"], batch["metadata"], strict=True):
            assert metadata == samples[tokens[0].item()].train_metadata
