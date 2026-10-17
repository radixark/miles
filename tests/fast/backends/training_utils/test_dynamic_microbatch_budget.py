from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import rollout


@pytest.mark.parametrize(
    "lengths,dp_microbatches,vpp_size,group_size,expected_count",
    [
        ([15, 52, 93, 39, 68, 59, 33, 28], 4, 1, 1, 4),
        ([15, 52, 93, 39, 68, 59, 33, 28], 6, 1, 1, 6),
        ([80, 80, 70, 20], 3, 2, 2, 4),
    ],
)
def test_dynamic_microbatches_preserve_token_budget(
    monkeypatch, lengths, dp_microbatches, vpp_size, group_size, expected_count
):
    state = SimpleNamespace(
        effective_dp=SimpleNamespace(size=1, group=None),
        cp=SimpleNamespace(size=1),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=group_size,
    )
    monkeypatch.setattr(rollout, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        rollout.GeneralPGUtil,
        "create",
        lambda _: SimpleNamespace(all_reduce=lambda tensor, *args, **kwargs: tensor.fill_(dp_microbatches)),
    )
    args = SimpleNamespace(
        use_dynamic_global_batch_size=True,
        global_batch_size=len(lengths),
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100,
        qkv_format="thd",
    )
    slots = [i % 2 for i in range(len(lengths))]
    data = dict(total_lengths=lengths, dynamic_global_batch_size=len(lengths), adapter_slots=slots)

    iterators, counts = rollout.get_data_iterator(args, [], data)

    assert counts == [expected_count]
    assert len(iterators) == vpp_size
    for iterator in iterators:
        bins = iterator.micro_batch_indices
        assert sorted(i for batch in bins for i in batch) == list(range(len(lengths)))
        assert all(batch and sum(lengths[i] for i in batch) <= 100 for batch in bins)
        assert all([slots[i] for i in batch] == sorted(slots[i] for i in batch) for batch in bins)
