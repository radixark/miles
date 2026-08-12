"""Distributed correctness test for the DeepSeek-V4 packed-THD Context Parallel layout.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/precision/test_dsv4_thd_cp_correctness.py
    torchrun --nproc_per_node=4 tests/e2e/precision/test_dsv4_thd_cp_correctness.py

A compressed group can straddle a CP split, so each rank pulls the rows its left neighbour
owns, compacts its own groups into fixed-capacity slots, and all-gathers. That decides where
rows land, not what is computed, so every check here is exact rather than a tolerance. Each
row carries its own global index, which makes a row fetched from the wrong place visible.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn  # cp_utils reaches for this submodule without importing it

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp
from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    compressor_boundary_width,
    exchange_cp_boundary_hidden,
)

register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["precision", "megatron"])
register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["precision"])

SEGMENTS = [1500, 2093, 500, 3]  # neither ratio divides these; the last is shorter than both
RATIOS = (4, 128)  # one layer that overlap-transforms, one that does not
WIDTH = 8


def setup_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size


def _stream(device):
    """Row t holds t + 1, leaving 0 free to mean "nothing here"."""
    rows = torch.arange(1, sum(SEGMENTS) + 1, device=device, dtype=torch.float32)
    return rows.view(-1, 1, 1).expand(-1, 1, WIDTH).contiguous()


def _group_starts(ratio):
    """First token of every compressed group, in sequence order, one segment at a time."""
    starts = [0] + torch.tensor(SEGMENTS).cumsum(0).tolist()
    return [starts[s] + c * ratio for s, ln in enumerate(SEGMENTS) for c in range(ln // ratio)]


def check_boundary_exchange(rank, world_size, ratio, x, l_local):
    """Forward delivers the left neighbour's tail; backward returns its gradient."""
    d_comp = compressor_boundary_width(ratio)
    start = rank * l_local
    local = x[start : start + l_local].clone().requires_grad_(True)
    boundary = exchange_cp_boundary_hidden(local, ratio=ratio, cp_group=dist.group.WORLD)

    want = torch.zeros_like(boundary) if rank == 0 else x[start - d_comp : start]
    forward_ok = torch.equal(boundary, want)

    # Weight by row, so a permuted boundary changes the gradient and not just its sum.
    weights = torch.arange(1, d_comp + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    (boundary * weights).sum().backward()
    grads = [torch.empty_like(local.grad) for _ in range(world_size)]
    dist.all_gather(grads, local.grad.contiguous())

    want_grad = torch.zeros_like(x)
    for r in range(1, world_size):
        want_grad[r * l_local - d_comp : r * l_local] = weights
    backward_ok = torch.equal(torch.cat(grads, dim=0), want_grad)
    return forward_ok, backward_ok


def check_compaction_and_layout(rank, world_size, ratio, x, l_local):
    """Every group is built from its own tokens, and stays reachable after the gather."""
    device = x.device
    cu = torch.tensor([0] + torch.tensor(SEGMENTS).cumsum(0).tolist(), dtype=torch.int32, device=device)
    start = rank * l_local
    c_cap = compact_group_capacity(l_local, ratio)
    local = x[start : start + l_local].clone().requires_grad_(True)
    boundary = exchange_cp_boundary_hidden(local, ratio=ratio, cp_group=dist.group.WORLD)
    compact, comp_ids = CompressorInputCompact.apply(local, boundary, cu, start, ratio, c_cap)

    slots = compact.view(c_cap, ratio, 1, WIDTH)
    live = comp_ids >= 0
    # A slot holds `ratio` consecutive tokens, and its first row says where the group starts.
    heads = slots[:, 0, 0, 0]
    steps = torch.arange(ratio, device=device, dtype=x.dtype).view(1, -1)
    want_slots = (heads.view(-1, 1) + steps).view(c_cap, ratio, 1, 1).expand_as(slots)
    contiguous_ok = torch.equal(slots[live], want_slots[live])
    padding_ok = not slots[~live].any()

    gathered = all_gather_cp(slots[:, 0].transpose(0, 1).contiguous(), dim=1, cp_group=dist.group.WORLD)
    mapping = compressed_rank_layout(
        cu, compressed_cu_seqlens(cu, ratio), l_local=l_local, cp_size=world_size, ratio=ratio, c_cap=c_cap
    )
    want_heads = torch.tensor(_group_starts(ratio), device=device, dtype=x.dtype) + 1
    layout_ok = torch.equal(gathered[0, mapping[: want_heads.numel()].long(), 0], want_heads)

    # Compaction is a gather, so each token pulled gets its gradient back exactly once -- twice
    # for the boundary group, which its owner and its left neighbour both build.
    compact.sum().backward()
    want_grad = torch.zeros(sum(SEGMENTS), device=device, dtype=x.dtype)
    for head in gathered[0, :, 0].tolist():
        if head:
            want_grad[int(head) - 1 : int(head) - 1 + ratio] += 1.0
    grads = [torch.empty_like(local.grad) for _ in range(world_size)]
    dist.all_gather(grads, local.grad.contiguous())
    grad_ok = torch.equal(torch.cat(grads, dim=0), want_grad.view(-1, 1, 1).expand(-1, 1, WIDTH))
    return contiguous_ok, padding_ok, layout_ok, grad_ok


def main():
    rank, world_size = setup_dist()
    try:
        assert sum(SEGMENTS) % world_size == 0
        l_local = sum(SEGMENTS) // world_size
        x = _stream(torch.cuda.current_device())
        passed = True
        for ratio in RATIOS:
            results = {}
            results["boundary forward"], results["boundary backward"] = check_boundary_exchange(
                rank, world_size, ratio, x, l_local
            )
            (
                results["group tokens"],
                results["capacity padding"],
                results["all-gather layout"],
                results["compaction backward"],
            ) = check_compaction_and_layout(rank, world_size, ratio, x, l_local)

            ok = all(results.values())
            passed = passed and ok
            if rank == 0:
                print(f"\n=== DeepSeek-V4 THD CP layout, compress_ratio={ratio}, CP={world_size} ===")
                for name, value in results.items():
                    print(f"{name:22s} PASS: {value}")
        if rank == 0:
            print("FAILED!" if not passed else f"\nCP={world_size} test PASSED!")
        if not passed:
            sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Self-bootstrap under torchrun when run as `python3 file.py`, the CUDA CI runner's mode.
    # Already inside torchrun => RANK is set.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
