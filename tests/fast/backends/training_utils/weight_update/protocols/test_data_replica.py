import torch

from miles.backends.training_utils.weight_update.protocols.utils.data_replica import OwnedBuckets


def _owned_names(replica_rank: int, buckets: list) -> list[str]:
    owned = OwnedBuckets(replica_rank=replica_rank, replica_size=2)
    for bucket in buckets:
        owned.offer(list(bucket))
    return [name for _, bucket in owned.take() for name, _ in bucket]


def test_one_large_bucket_is_balanced_by_bytes_not_by_count():
    """Round-robin would give rank 0 the large bucket and half the small ones; bytes give it the large one alone."""
    buckets = [[("embed", torch.zeros(300))]] + [[(f"layer{i}", torch.zeros(100))] for i in range(3)]
    assert _owned_names(0, buckets) == ["embed"]
    assert _owned_names(1, buckets) == ["layer0", "layer1", "layer2"]


def test_an_offered_bucket_is_emptied_so_its_device_tensors_free_before_the_next_gather():
    """The caller's loop still holds the list while the next bucket is gathered."""
    owned = OwnedBuckets(replica_rank=1, replica_size=2)
    bucket = [("w", torch.zeros(4))]
    owned.offer(bucket)
    assert bucket == []


def test_an_owned_bucket_lands_in_one_host_buffer_with_every_tensor_aligned():
    """Shared memory holds a bucket as one buffer; ordered by dtype, widest first, no tensor lands misaligned."""
    bucket = [
        ("a.w", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)),
        ("a.scale", torch.tensor([0.5, 1.5])),
        ("b.w", torch.full((3,), 2.0, dtype=torch.bfloat16)),
        ("mask", torch.tensor([1, 0, 1], dtype=torch.int8)),
        ("empty", torch.empty(0, 2, dtype=torch.bfloat16)),
        ("b.scale", torch.tensor([-1.0])),
    ]
    owned = OwnedBuckets(replica_rank=0, replica_size=1)
    owned.offer(list(bucket))
    ((_, host_bucket),) = owned.take()

    assert [name for name, _ in host_bucket] == ["a.scale", "b.scale", "a.w", "b.w", "empty", "mask"]
    expected = dict(bucket)
    for name, tensor in host_bucket:
        assert tensor.dtype == expected[name].dtype and torch.equal(tensor, expected[name])
    storages = {tensor.untyped_storage().data_ptr() for _, tensor in host_bucket}
    assert len(storages) == 1, "the bucket is not one host buffer"
