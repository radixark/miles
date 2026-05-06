"""Distributed correctness test for zigzag <-> packed-shard hybrid CP relayout.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/precision/test_hf_attention_cp_relayout.py
    torchrun --nproc_per_node=4 tests/e2e/precision/test_hf_attention_cp_relayout.py
"""

import os
import sys

import torch
import torch.distributed as dist

from miles_plugins.models.hf_attention import _packed_shard_to_zigzag, _zigzag_to_packed_shard


def setup_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _make_subchunk(sample_id: int, sub_id: int, chunk_len: int, device: torch.device) -> torch.Tensor:
    base = sample_id * 1000 + sub_id * 100
    values = torch.arange(base, base + chunk_len, device=device, dtype=torch.float32)
    return values.view(-1, 1, 1)


def _build_rank_inputs(rank: int, world_size: int, device: torch.device):
    chunk_lens = [3, 5]
    tail_pad_local_len = 3
    zigzag_chunks = []
    full_sequences = []
    cu = [0]

    for sample_id, chunk_len in enumerate(chunk_lens):
        subchunks = [_make_subchunk(sample_id, sub_id, chunk_len, device) for sub_id in range(2 * world_size)]
        zigzag_chunks.extend([subchunks[rank], subchunks[2 * world_size - 1 - rank]])
        full_sequences.append(torch.cat(subchunks, dim=0))
        cu.append(cu[-1] + 2 * world_size * chunk_len)

    tail_pad = (rank * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
    zigzag_chunks.append(tail_pad)
    full_sequences.append(
        torch.cat(
            [
                (r * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
                for r in range(world_size)
            ],
            dim=0,
        )
    )
    cu.append(cu[-1] + world_size * tail_pad_local_len)

    zigzag = torch.cat(zigzag_chunks, dim=0).requires_grad_(True)
    packed_full = torch.cat(full_sequences, dim=0)
    local_len = zigzag.size(0)
    packed_shard = packed_full[rank * local_len : (rank + 1) * local_len]
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    return zigzag, packed_shard, cu_seqlens


def test_relayout(rank: int, world_size: int):
    device = torch.device(f"cuda:{rank}")
    cp_group = dist.group.WORLD

    zigzag, expected_packed_shard, cu_seqlens = _build_rank_inputs(rank, world_size, device)

    packed_shard = _zigzag_to_packed_shard(zigzag, cu_seqlens, cp_group, rank, world_size)
    roundtrip = _packed_shard_to_zigzag(packed_shard, cu_seqlens, cp_group, rank, world_size)

    packed_ok = torch.equal(packed_shard, expected_packed_shard)
    roundtrip_ok = torch.equal(roundtrip, zigzag)

    loss = roundtrip.sum()
    loss.backward()
    grad_ok = torch.equal(zigzag.grad, torch.ones_like(zigzag))

    passed = packed_ok and roundtrip_ok and grad_ok
    if rank == 0:
        print(f"\n=== HF Attention Hybrid CP Relayout Test CP={world_size} ===")
        print(f"zigzag->packed PASS: {packed_ok}")
        print(f"roundtrip PASS: {roundtrip_ok}")
        print(f"backward PASS: {grad_ok}")
        if not passed:
            sys.exit(1)


def main():
    rank, world_size, _ = setup_dist()
    try:
        test_relayout(rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
