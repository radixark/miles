from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import DataIterator, get_batch, prepare_rollout_data
from miles.platforms import current_platform
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp_raw,
)
from miles.utils.types import Sample


@pytest.mark.parametrize(
    "metadata",
    [
        [None, {}, None, {"target": 37}],
        [{"target": 13}, {"target": 17}, None, {}],
        [None, {}, None, {}],
        [None, None, None, None],
    ],
    ids=["first-missing", "first-present", "empty-only", "absent"],
)
def test_train_metadata_follows_its_sample_to_the_loss_batch(monkeypatch, metadata):
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        length_penalty_coeff=0,
        enable_sample_ownership_checker=False,
        use_output_store=False,
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_dynamic_global_batch_size=False,
        multimodal_token_space_processing=False,
        balance_data=False,
        enable_witness=False,
        qkv_format="thd",
    )
    samples = [
        Sample(index=i, tokens=[0, i + 1], response_length=1, reward=1.0, train_metadata=entry)
        for i, entry in enumerate(metadata)
    ]
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    present = any(entry is not None for entry in metadata)
    if present:
        assert data.get("metadata") == metadata
    else:
        assert "metadata" not in data
    group = parallel.GroupInfo(rank=0, size=1, group=None)
    state = parallel.ParallelState(
        **{k: group for k in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setattr(current_platform, "current_device", lambda: torch.device("cpu"))
    for rank, shard in enumerate(split_train_data_by_dp_raw(args, data, dp_size=2)):
        if present:
            assert shard.get("metadata") == metadata[rank::2]
        shard = prepare_rollout_data(args, process_rollout_data_shard(args, shard))
        iterator = DataIterator(shard, micro_batch_indices=[[1, 0]])
        batch = get_batch(iterator, ["tokens", "loss_masks", "total_lengths", "response_lengths", "sample_indices"])
        assert batch["sample_indices"] == [rank + 2, rank]
        if present:
            assert batch.get("metadata") == [metadata[rank + 2], metadata[rank]]
        else:
            assert "metadata" not in batch
