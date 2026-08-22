"""Check that ``pp_assemble_full_adapter`` reunites a pipeline-split adapter.

The exporter gathers TP and EP but not PP, so each stage only sees its own layers.
The weight-sync path has always assembled across PP; ``save_lora_checkpoint`` did not,
which meant a PP > 1 run wrote an ``adapter_model.safetensors`` holding stage 0's
layers alone, with no error.

    PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 2 \
        tests/manual/lora/verify_pp_adapter_assembly.py

Exits nonzero if any check fails.
"""

import os
import sys

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora_utils import pp_assemble_full_adapter
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo

FAILS: list[str] = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    if dist.get_rank() == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def trivial(group=None) -> GroupInfo:
    return GroupInfo(rank=0, size=1, group=group)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    assert world >= 2, "needs at least 2 ranks to have a pipeline"

    pp_group = dist.new_group(list(range(world)))
    set_parallel_state(
        ParallelState(
            intra_dp=trivial(),
            intra_dp_cp=trivial(),
            cp=trivial(),
            tp=trivial(),
            pp=GroupInfo(rank=rank, size=world, group=pp_group),
            ep=trivial(),
            etp=trivial(),
            indep_dp=trivial(),
        )
    )

    layers_per_stage = 3
    local = [
        (
            f"model.layers.{rank * layers_per_stage + i}.self_attn.q_proj.lora_A.weight",
            torch.full((4, 8), float(rank * layers_per_stage + i), device="cuda", dtype=torch.bfloat16),
        )
        for i in range(layers_per_stage)
    ]
    assembled = pp_assemble_full_adapter(local)
    names = [n for n, _ in assembled]

    expected = sorted(
        f"model.layers.{stage * layers_per_stage + i}.self_attn.q_proj.lora_A.weight"
        for stage in range(world)
        for i in range(layers_per_stage)
    )
    check(
        f"every stage's layers present on rank {rank}",
        names == expected,
        f"n={len(names)} (expect {len(expected)})",
    )

    by_name = dict(assembled)
    values_ok = all(
        by_name[f"model.layers.{layer}.self_attn.q_proj.lora_A.weight"].float().eq(float(layer)).all().item()
        for layer in range(world * layers_per_stage)
    )
    check("each layer carries its own stage's values", values_ok)

    other_stage = f"model.layers.{((rank + 1) % world) * layers_per_stage}.self_attn.q_proj.lora_A.weight"
    check(f"rank {rank} received a layer it does not own", other_stage in by_name)

    dist.barrier()
    if rank == 0:
        print(("\n=== PP ASSEMBLY: ALL PASS ===" if not FAILS else f"\n=== FAILED: {FAILS} ==="), flush=True)
    dist.destroy_process_group()
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
