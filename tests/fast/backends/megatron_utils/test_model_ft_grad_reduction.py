from types import SimpleNamespace
from typing import Any

import torch

from miles.backends.megatron_utils.model import _run_nominal_dp_shards


def test_nominal_dp_shards_are_reduced_separately_then_combined() -> None:
    """A degraded retry must reproduce the nominal DP shard reduction tree."""
    buckets = [
        SimpleNamespace(gradient_scaling_factor=0.5, grad_data=torch.zeros(2)),
        SimpleNamespace(gradient_scaling_factor=0.5, grad_data=torch.zeros(1)),
    ]
    zero_calls: list[bool] = []

    def zero_grad_buffer() -> None:
        zero_calls.append(True)
        for bucket in buckets:
            bucket.grad_data.zero_()

    model = [
        SimpleNamespace(
            bucket_groups=[SimpleNamespace(buckets=[buckets[0]])],
            expert_parallel_bucket_groups=[SimpleNamespace(buckets=[buckets[1]])],
            zero_grad_buffer=zero_grad_buffer,
        )
    ]
    raw_shard_grads = [
        [torch.tensor([2.0, 4.0]), torch.tensor([10.0])],
        [torch.tensor([6.0, 8.0]), torch.tensor([14.0])],
    ]
    calls: list[dict[str, Any]] = []

    def forward_backward_func(**kwargs: Any) -> list[dict[str, torch.Tensor | list[str]]]:
        shard_grads = raw_shard_grads[len(calls)]
        calls.append(kwargs)
        for bucket, grad in zip(buckets, shard_grads, strict=True):
            bucket.grad_data.copy_(grad * bucket.gradient_scaling_factor)
        return [{"keys": ["loss"], "values": torch.tensor([1.0])}]

    losses = _run_nominal_dp_shards(
        args=SimpleNamespace(
            ci_inject_rollout_data_group_by_dp_size=2,
            seq_length=128,
            micro_batch_size=128,
            decoder_seq_length=None,
        ),
        data_iterator=[iter(())],
        model=model,
        forward_step=lambda: None,
        forward_backward_func=forward_backward_func,
    )

    assert len(losses) == 2
    assert len(calls) == 2
    assert all(call["num_microbatches"] == 1 for call in calls)
    assert all(call["force_all_reduce"] is True for call in calls)
    assert zero_calls == [True]
    assert torch.equal(buckets[0].grad_data, torch.tensor([2.0, 3.0]))
    assert torch.equal(buckets[1].grad_data, torch.tensor([6.0]))
    assert [bucket.gradient_scaling_factor for bucket in buckets] == [0.5, 0.5]
