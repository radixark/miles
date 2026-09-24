"""Speed, memory and precision of the head-sharded delta-rule layer at the attention configs of the
latest models, across sequence lengths and one TP x CP layout (world size = TP * CP).

    torchrun --nproc_per_node=<TP*CP> tests/manual/bench_delta_rule.py <TP> <CP> [seq_len ...]

Per (model, seq_len): wall ms / peak GiB of fwd+bwd for the head-sharded layer (sequence parallel,
zigzag CP relayout as in training) and, at CP=1, the replicated layer it replaced; relative error of
the output and input gradient against the replicated layer (CP=1) or against the same layer without
CP (CP>1, contiguous shards).
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
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.kernels.attention.delta_rule import DeltaRuleHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast-gpu"))
from delta_rule_reference import ReplicatedGDN, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

MODELS = {
    "qwen3.8-flash GDN": (2560, DeltaRuleHeads(16, 48, 128, 128), "gdn"),
    "glm5.3-flash KDA": (4096, DeltaRuleHeads(64, 64, 128, 128), "kda"),
    "kimi-k3 KDA": (7168, DeltaRuleHeads(96, 96, 128, 128), "kda"),
}


def wall_ms(fn, iters):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3, torch.cuda.max_memory_allocated() / 2**30


def zigzag_shard(full, cp_rank, cp_size):
    chunks = full.chunk(2 * cp_size, dim=0)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]]).contiguous()


def contiguous_shard(full, cp_rank, cp_size):
    return full.chunk(cp_size, dim=0)[cp_rank].contiguous()


def tp_gather(x, tp_group):
    return (
        gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
        if tp_group.size() > 1
        else x
    )


def sp_shard(x, tp_group):
    return x.chunk(tp_group.size(), dim=0)[tp_group.rank()].contiguous()


def run(name, seq_len, tp, cp, log):
    hidden, heads, kind = MODELS[name]
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads.num_v_heads,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=tp > 1,
    )
    torch.manual_seed(7)
    ref = (
        ReplicatedKDA(hidden, heads, torch.bfloat16)
        if kind == "kda"
        else ReplicatedGDN("qwen3_5", hidden, heads, torch.bfloat16)
    ).cuda()
    torch.manual_seed(11)
    full = torch.randn(seq_len, 1, hidden, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(full)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    iters = max(2, min(10, 262144 // seq_len))
    row = f"[{name:18s} TP={tp} CP={cp} seq={seq_len:6d}]"
    try:
        layer = build_layer(ref, config, allgather_cp=False)
        x = sp_shard(zigzag_shard(full, cp_group.rank(), cp), tp_group).requires_grad_()
        new_ms, new_mem = wall_ms(lambda: layer(x, packed_seq_params=packed(cu))[0].float().sum().backward(), iters)
        row += f"  head-sharded {new_ms:8.2f} ms {new_mem:6.2f} GiB"

        if cp == 1:

            def replicated(x=x):
                whole = (
                    gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
                    if tp > 1
                    else x
                )
                out = ref(whole.transpose(0, 1), cu).transpose(0, 1)
                return scatter_to_sequence_parallel_region(out, group=tp_group) if tp > 1 else out

            old_ms, old_mem = wall_ms(lambda: replicated().float().sum().backward(), iters)
            row += f"  replicated {old_ms:8.2f} ms {old_mem:6.2f} GiB  speed x{old_ms / new_ms:4.2f} mem x{old_mem / new_mem:4.2f}"
            x_old, x_new = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
            g = sp_shard(grad, tp_group)
            out_old = replicated(x_old)
            out_old.backward(g)
            out_new = layer(x_new, packed_seq_params=packed(cu))[0]
            out_new.backward(g)
            row += f"  rel err out {rel_err(out_old, out_new):.1e} dx {rel_err(x_old.grad, x_new.grad):.1e}"
        else:
            contig = build_layer(ref, config, allgather_cp=True)
            x_cp = sp_shard(contiguous_shard(full, cp_group.rank(), cp), tp_group).requires_grad_()
            out_cp = contig(x_cp, packed_seq_params=packed(cu))[0]
            out_cp.backward(sp_shard(contiguous_shard(grad, cp_group.rank(), cp), tp_group))
            core = contig.linear_attn
            x_ref = sp_shard(full, tp_group).requires_grad_()
            whole = (
                gather_from_sequence_parallel_region(x_ref, tensor_parallel_output_grad=True, group=tp_group)
                if tp > 1
                else x_ref
            )
            out_ref = core.out_proj(core(whole.transpose(0, 1), cu, None)).transpose(0, 1)
            out_ref = (
                reduce_scatter_to_sequence_parallel_region(out_ref, group=tp_group)
                if tp > 1
                else reduce_from_tensor_model_parallel_region(out_ref, group=tp_group)
            )
            out_ref.backward(sp_shard(grad, tp_group))
            start = cp_group.rank() * (seq_len // cp)
            ref_out, ref_dx = tp_gather(out_ref.detach(), tp_group), tp_gather(x_ref.grad, tp_group)
            cp_out, cp_dx = tp_gather(out_cp.detach(), tp_group), tp_gather(x_cp.grad, tp_group)
            window = slice(start, start + seq_len // cp)
            row += f"  vs no-CP rel err out {rel_err(ref_out[window], cp_out):.1e} dx {rel_err(ref_dx[window], cp_dx):.1e}"
    except torch.cuda.OutOfMemoryError:
        row += "  OOM"
    torch.cuda.empty_cache()
    log(row)


def main():
    tp, cp = int(sys.argv[1]), int(sys.argv[2])
    seq_lens = [int(s) for s in sys.argv[3:]] or [4096, 8192, 16384, 32768, 65536, 131072]
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)
    log = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    for name in MODELS:
        for seq_len in seq_lens:
            run(name, seq_len, tp, cp, log)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
