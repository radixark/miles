"""Distributed correctness test for zigzag <-> packed-shard hybrid CP relayout.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/precision/test_hf_attention_cp_relayout.py
    torchrun --nproc_per_node=4 tests/e2e/precision/test_hf_attention_cp_relayout.py
"""

import os

import torch
import torch.distributed as dist

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=30, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=4
)
register_rocm_ci(est_time=30, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])

from miles_plugins.models.cp_layout import build_cp_layout_metadata
from miles_plugins.models.cp_utils import packed_shard_to_zigzag, zigzag_to_packed_shard


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


def test_relayout(rank: int, world_size: int, cp_group):
    device = torch.device("cuda", torch.cuda.current_device())

    zigzag, expected_packed_shard, cu_seqlens = _build_rank_inputs(rank, world_size, device)
    layout = build_cp_layout_metadata(cu_seqlens.cpu().tolist(), world_size, rank)

    for communication in ("all_gather", "all_to_all"):
        zigzag.grad = None
        packed_shard = zigzag_to_packed_shard(
            zigzag, None, cp_group, rank, world_size, layout=layout, communication=communication
        )
        torch.testing.assert_close(packed_shard, expected_packed_shard, atol=0, rtol=0)
        (packed_shard * expected_packed_shard).sum().backward()
        torch.testing.assert_close(zigzag.grad, zigzag.detach(), atol=0, rtol=0)

        packed_leaf = expected_packed_shard.clone().requires_grad_(True)
        roundtrip = packed_shard_to_zigzag(
            packed_leaf, None, cp_group, rank, world_size, layout=layout, communication=communication
        )
        torch.testing.assert_close(roundtrip, zigzag.detach(), atol=0, rtol=0)
        (roundtrip * zigzag.detach()).sum().backward()
        torch.testing.assert_close(packed_leaf.grad, expected_packed_shard, atol=0, rtol=0)

    if rank == 0:
        print(f"CP={world_size}: planned all-gather/all-to-all layout, inverse and gradients PASS")


def test_against_unplanned_relayout(rank, cp_size, cp_group):
    device = torch.device("cuda", torch.cuda.current_device())
    for boundaries in ([0, cp_size], [0, 8 * cp_size, 10 * cp_size, 13 * cp_size]):
        layout = build_cp_layout_metadata(boundaries, cp_size, rank)
        cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(17 + rank)
            hidden = torch.randn(layout.local_total_len, 3, 2, device=device, dtype=dtype).transpose(1, 2)
            grad_output = torch.randn_like(hidden.transpose(1, 2)).transpose(1, 2)
            for transform in (zigzag_to_packed_shard, packed_shard_to_zigzag):
                reference_input = hidden.detach().requires_grad_(True)
                reference = transform(reference_input, cu_seqlens, cp_group, rank, cp_size)
                reference.backward(grad_output)
                for communication in ("all_gather", "all_to_all"):
                    candidate_input = hidden.detach().requires_grad_(True)
                    candidate = transform(
                        candidate_input, None, cp_group, rank, cp_size, layout=layout, communication=communication
                    )
                    candidate.backward(grad_output)
                    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
                    torch.testing.assert_close(candidate_input.grad, reference_input.grad, atol=0, rtol=0)
    if rank == 0:
        print(f"CP={cp_size}: FP32/BF16, noncontiguous tensors, zero splits, forward/backward match main exactly")


def main():
    rank, world_size, _ = setup_dist()
    try:
        for cp_size in sorted({1, 2, world_size}):
            for start in range(0, world_size, cp_size):
                ranks = list(range(start, start + cp_size))
                cp_group = dist.new_group(ranks)
                if rank in ranks:
                    test_relayout(rank - start, cp_size, cp_group)
                    test_against_unplanned_relayout(rank - start, cp_size, cp_group)
                    dist.destroy_process_group(cp_group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Self-bootstrap under torchrun when invoked as `python3 file.py` (the
    # CUDA CI runner's mode). Already inside torchrun => RANK is set.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
