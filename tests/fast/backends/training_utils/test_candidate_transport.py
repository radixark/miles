from argparse import Namespace

import torch

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw


def test_dp_partition_keeps_candidate_support_and_teacher_paired():
    fields = {
        "opd_candidate_ids": [torch.tensor([[100 + i, 200 + i]]) for i in range(4)],
        "opd_candidate_old_log_probs": [torch.tensor([[-i - 1.0, -i - 2.0]]) for i in range(4)],
        "opd_candidate_teacher_log_probs": [torch.tensor([[-i - 3.0, -i - 4.0]]) for i in range(4)],
        "opd_loss_weights": [torch.tensor([i + 1.0]) for i in range(4)],
    }
    data = dict(tokens=[[i, i + 1] for i in range(4)], response_lengths=[1] * 4, loss_masks=[[1]] * 4, **fields)
    shards = split_train_data_by_dp_raw(Namespace(balance_data=False), data, dp_size=2)
    for rank, shard in enumerate(shards):
        for key, values in fields.items():
            for actual, index in zip(shard[key], [rank, rank + 2], strict=True):
                torch.testing.assert_close(actual, values[index])
