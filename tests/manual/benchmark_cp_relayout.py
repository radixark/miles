"""CUDA microbenchmark of CP relayout forward/backward, not end-to-end training throughput.

torchrun --standalone --nproc-per-node=4 tests/manual/benchmark_cp_relayout.py --output /tmp/cp4.json
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from miles_plugins.models.cp_utils import packed_shard_to_zigzag, zigzag_to_packed_shard


def measure(hidden, grad_output, cu_seqlens, mode, warmup, iterations):
    rank, cp_size = dist.get_rank(), dist.get_world_size()
    communication = "all_gather" if mode == "main" else "all_to_all"

    def step():
        hidden.grad = None
        packed = zigzag_to_packed_shard(
            hidden, cu_seqlens, dist.group.WORLD, rank, cp_size, communication=communication
        )
        output = packed_shard_to_zigzag(
            packed, cu_seqlens, dist.group.WORLD, rank, cp_size, communication=communication
        )
        output.backward(grad_output)

    for _ in range(warmup):
        step()
    hidden.grad = None
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        step()
    end_event.record()
    torch.cuda.synchronize()
    return {
        "rank": rank,
        "cuda_ms": start_event.elapsed_time(end_event) / iterations,
        "wall_ms": (time.perf_counter() - wall_start) * 1000 / iterations,
        "incremental_peak_mib": (torch.cuda.max_memory_allocated() - allocated_before) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, cp_size = dist.get_rank(), dist.get_world_size()
    results = []
    try:
        for tokens in args.tokens:
            if tokens % (8 * cp_size):
                raise ValueError("Token count must be divisible by 8 * CP")
            boundaries = [0, tokens // 4, tokens // 2, 3 * tokens // 4, tokens, tokens + 3 * cp_size]
            cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
            hidden = torch.randn(boundaries[-1] // cp_size, 1, args.hidden_size, device="cuda", dtype=torch.bfloat16)
            hidden.requires_grad_(True)
            grad_output = torch.randn_like(hidden)
            for mode in ("main", "all_to_all"):
                metrics = measure(hidden, grad_output, cu_seqlens, mode, args.warmup, args.iterations)
                per_rank = [None] * cp_size
                dist.all_gather_object(per_rank, metrics)
                row = {
                    "global_tokens": boundaries[-1],
                    "mode": mode,
                    "ranks": per_rank,
                    "max_cuda_ms": max(item["cuda_ms"] for item in per_rank),
                    "median_cuda_ms": statistics.median(item["cuda_ms"] for item in per_rank),
                    "max_incremental_peak_mib": max(item["incremental_peak_mib"] for item in per_rank),
                }
                results.append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
            del hidden, grad_output
        if rank == 0:
            args.output.write_text(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(),
                        "cp_size": cp_size,
                        "hidden_size": args.hidden_size,
                        "dtype": "bfloat16",
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
