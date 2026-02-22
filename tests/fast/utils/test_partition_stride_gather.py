"""
Unit tests for the partition_stride > 1 all-gather reconstruction logic in
miles/backends/megatron_utils/update_weight/common.py.

These tests run without Megatron or distributed dependencies by replicating the
sharding/reconstruction logic in pure PyTorch.

Megatron sharding with partition_stride=S, tp_size=N:
  - Split full tensor into N*S equal chunks along partition_dim.
  - Rank r stores chunks [r, r+N, r+2N, ..., r+(S-1)*N] concatenated (contiguous).

Reconstruction (our implementation):
  - Split each rank's tensor back into S chunks.
  - Interleave: [rank0_s0, rank1_s0, ..., rankN_s0, rank0_s1, ..., rankN_s{S-1}]
  - torch.cat along partition_dim -> original tensor.
"""

import pytest
import torch


def _megatron_shard(
    full_tensor: torch.Tensor, tp_size: int, partition_dim: int, partition_stride: int
) -> list[torch.Tensor]:
    """
    Simulate how Megatron shards a tensor with partition_stride.
    Returns a list of per-rank tensors (stored contiguously).
    """
    # Split into tp_size * partition_stride equal chunks
    chunks = full_tensor.chunk(tp_size * partition_stride, dim=partition_dim)
    assert (
        len(chunks) == tp_size * partition_stride
    ), f"tensor size along dim {partition_dim} must be divisible by tp_size*stride"

    # Rank r gets chunks [r, r+tp_size, r+2*tp_size, ...]
    per_rank = []
    for r in range(tp_size):
        rank_chunks = [chunks[r + s * tp_size] for s in range(partition_stride)]
        per_rank.append(torch.cat(rank_chunks, dim=partition_dim))
    return per_rank


def _reconstruct(param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int) -> torch.Tensor:
    """
    The reconstruction logic from common.py (copied verbatim for isolation).
    """
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    else:
        chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
        interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
        return torch.cat(interleaved, dim=partition_dim)


# ---------------------------------------------------------------------------
# stride == 1 (baseline, existing behaviour)
# ---------------------------------------------------------------------------


class TestPartitionStrideOne:
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("partition_dim", [0, 1])
    def test_round_trip(self, tp_size, partition_dim):
        shape = [32, 64]
        full = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        partitions = _megatron_shard(full, tp_size, partition_dim, partition_stride=1)
        reconstructed = _reconstruct(partitions, partition_dim, partition_stride=1)
        assert torch.equal(reconstructed, full), f"Mismatch for tp_size={tp_size} partition_dim={partition_dim}"


# ---------------------------------------------------------------------------
# stride > 1  (the newly supported case)
# ---------------------------------------------------------------------------


class TestPartitionStrideGreaterThanOne:
    @pytest.mark.parametrize(
        "tp_size,partition_stride",
        [
            (2, 2),
            (4, 2),
            (4, 4),
            (8, 2),
            (2, 3),
            (4, 3),
        ],
    )
    @pytest.mark.parametrize("partition_dim", [0, 1])
    def test_round_trip(self, tp_size, partition_stride, partition_dim):
        """Shard then reconstruct must recover the original tensor exactly."""
        # Size must be divisible by tp_size * partition_stride
        size = tp_size * partition_stride * 8  # 8 elements per leaf chunk
        shape = [size, 64] if partition_dim == 0 else [64, size]
        full = torch.randn(shape)
        partitions = _megatron_shard(full, tp_size, partition_dim, partition_stride)
        reconstructed = _reconstruct(partitions, partition_dim, partition_stride)
        assert reconstructed.shape == full.shape, f"Shape mismatch: {reconstructed.shape} != {full.shape}"
        assert torch.equal(
            reconstructed, full
        ), f"Value mismatch for tp_size={tp_size} stride={partition_stride} dim={partition_dim}"

    @pytest.mark.parametrize("tp_size,partition_stride", [(2, 2), (4, 2), (4, 4)])
    def test_each_rank_gets_correct_rows(self, tp_size, partition_stride):
        """Verify that each rank holds the expected non-contiguous rows from the full weight."""
        size = tp_size * partition_stride * 4
        full = torch.arange(size, dtype=torch.float32).unsqueeze(1).expand(size, 8).contiguous()
        # full[:, 0] == [0, 1, 2, ..., size-1], one unique id per row
        partitions = _megatron_shard(full, tp_size, 0, partition_stride)
        chunk_size = size // (tp_size * partition_stride)
        for r, part in enumerate(partitions):
            expected_ids = []
            for s in range(partition_stride):
                start = (r + s * tp_size) * chunk_size
                expected_ids.extend(range(start, start + chunk_size))
            actual_ids = part[:, 0].long().tolist()
            assert actual_ids == expected_ids, f"Rank {r} holds wrong rows: expected {expected_ids}, got {actual_ids}"

    def test_qkv_gqa_shape(self):
        """
        Realistic GQA QKV scenario: Qwen3-like model, tp=4, stride=3 (Q+K+V heads).
        Verifies shape and value correctness.
        """
        num_heads = 32
        head_dim = 128
        tp_size = 4
        partition_stride = 3  # Q, K, V groups
        # full qkv weight: [num_heads * 3 * head_dim, hidden]  (simplified)
        full_size = num_heads * partition_stride * head_dim
        hidden = 256
        full = torch.randn(full_size, hidden)
        partitions = _megatron_shard(full, tp_size, 0, partition_stride)
        reconstructed = _reconstruct(partitions, 0, partition_stride)
        assert reconstructed.shape == full.shape
        assert torch.equal(reconstructed, full)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_rank_any_stride(self):
        """With tp_size=1, reconstruction is always just the tensor itself."""
        for stride in [1, 2, 4]:
            full = torch.randn(24, 16)
            partitions = _megatron_shard(full, 1, 0, stride)
            assert len(partitions) == 1
            reconstructed = _reconstruct(partitions, 0, stride)
            assert torch.equal(reconstructed, full)

    def test_stride_equals_tp_size(self):
        """stride == tp_size is a valid extreme case."""
        tp_size = 4
        stride = 4
        size = tp_size * stride * 2
        full = torch.randn(size, 32)
        partitions = _megatron_shard(full, tp_size, 0, stride)
        reconstructed = _reconstruct(partitions, 0, stride)
        assert torch.equal(reconstructed, full)

    def test_3d_tensor_partition_dim_0(self):
        """3-D weight (e.g. grouped matmul expert weights) along dim 0."""
        tp_size, stride = 4, 2
        size = tp_size * stride * 4
        full = torch.randn(size, 16, 8)
        partitions = _megatron_shard(full, tp_size, 0, stride)
        reconstructed = _reconstruct(partitions, 0, stride)
        assert torch.equal(reconstructed, full)

    def test_stride_1_vs_stride_gt1_same_result_when_stride1(self):
        """Ensures stride=1 path and generic path agree when stride==1."""
        tp_size = 4
        full = torch.randn(32, 64)
        partitions = _megatron_shard(full, tp_size, 0, 1)
        r1 = _reconstruct(partitions, 0, partition_stride=1)
        # Force through the stride>1 code path with stride=1 equivalent manual interleave
        chunks_per_rank = [p.chunk(1, dim=0) for p in partitions]
        interleaved = [chunks_per_rank[r][0] for r in range(tp_size)]
        r2 = torch.cat(interleaved, dim=0)
        assert torch.equal(r1, r2)
