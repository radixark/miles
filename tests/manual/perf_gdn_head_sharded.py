"""One GDN layer, fwd+bwd wall time, GPU time and peak memory: replicated (pre-sharding) vs head-sharded.

    torchrun --nproc_per_node=2 tests/manual/perf_gdn_head_sharded.py <sp|nosp> <hidden> <k_heads> <v_heads> <seq_len>...

With ``sp`` both run as they do under ``--sequence-parallel``: the replicated layer all-gathers its input
and slices its output (what the old HF-attention shim did), the head-sharded layer takes the SP shard.
"""

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.profiler import ProfilerActivity, profile

from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast-gpu"))
from delta_rule_reference import ReplicatedGDN, build_layer, packed  # noqa: E402


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    gpu = sum(e.self_device_time_total for e in prof.key_averages()) / iters / 1e3
    return wall, gpu, torch.cuda.max_memory_allocated() / 2**30


def main():
    sequence_parallel = sys.argv[1] == "sp"
    hidden, k_heads, v_heads = (int(a) for a in sys.argv[2:5])
    seq_lens = [int(a) for a in sys.argv[5:]]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=16,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
        sequence_parallel=sequence_parallel,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    heads = DeltaRuleHeads(k_heads, v_heads, 128, 128)
    ref = ReplicatedGDN("qwen3_5", hidden, heads, torch.bfloat16).cuda()
    layer = build_layer(ref, config)
    if rank == 0:
        print(
            f"GDN layer fwd+bwd, hidden={hidden} heads={k_heads}/{v_heads}, TP={world}, SP={sequence_parallel}; wall ms / GPU ms / peak GiB"
        )
    for seq_len in seq_lens:
        local_len = seq_len // world if sequence_parallel else seq_len
        x_sbh = torch.randn(local_len, 1, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

        def replicated(x=x_sbh, cu=cu):
            full = (
                gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
                if sequence_parallel
                else x
            )
            out = ref(full.transpose(0, 1), cu).transpose(0, 1)
            out = scatter_to_sequence_parallel_region(out, group=tp_group) if sequence_parallel else out
            out.float().sum().backward()

        old = bench(replicated)
        new = bench(lambda x=x_sbh, cu=cu: layer(x, packed_seq_params=packed(cu))[0].float().sum().backward())
        if rank == 0:
            print(
                f"  seq {seq_len:6d}  replicated {old[0]:7.2f} / {old[1]:7.2f} / {old[2]:5.2f}"
                f"   head-sharded {new[0]:7.2f} / {new[1]:7.2f} / {new[2]:5.2f}"
                f"   wall x{old[0] / new[0]:.2f}  gpu x{old[1] / new[1]:.2f}  mem x{old[2] / new[2]:.2f}"
            )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__, *sys.argv[1:]])
    main()
