import torch

from miles.backends.training_utils.weight_update.protocols.utils.data_replica import OwnedBuckets


def _owned_names(replica_rank: int, buckets: list) -> list[str]:
    owned = OwnedBuckets(replica_rank=replica_rank, replica_size=2)
    for bucket in buckets:
        owned.offer(bucket)
    return [name for _, bucket in owned.take() for name, _ in bucket]


def test_one_large_bucket_is_balanced_by_bytes_not_by_count():
    """Round-robin would give rank 0 the large bucket and half the small ones; bytes give it the large one alone."""
    buckets = [[("embed", torch.zeros(300))]] + [[(f"layer{i}", torch.zeros(100))] for i in range(3)]
    assert _owned_names(0, buckets) == ["embed"]
    assert _owned_names(1, buckets) == ["layer0", "layer1", "layer2"]
