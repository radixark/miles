import pytest
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


def test_a_failed_copy_out_is_raised_after_the_stream_not_mid_gather():
    """The gathers are collective: a rank leaving the stream early would leave the others waiting in them."""

    def full(index, bucket):
        raise OSError(28, "No space left on device")

    owned = OwnedBuckets(replica_rank=0, replica_size=1, host_bucket=full)
    buckets = [[(f"layer{i}", torch.zeros(4))] for i in range(3)]
    for bucket in buckets:
        owned.offer(bucket)  # each bucket of the stream is still consumed

    assert all(bucket == [] for bucket in buckets)
    with pytest.raises(OSError, match="No space left"):
        owned.take()
