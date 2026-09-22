"""Head-sharded GDN and KDA under native fla context parallelism against the same layer on the whole
sequence. The sequence boundary falls inside a CP shard, so the recurrent state is passed mid-sequence.

    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=2
    torchrun --nproc_per_node=4 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=4
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fast-gpu"))
from delta_rule_reference import ReplicatedGDN, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=120, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])

HIDDEN = 256


def check(ref, config, rank, world_size):
    layer = build_layer(ref, config, allgather_cp=True)
    core = layer.linear_attn
    total = 128 * world_size
    torch.manual_seed(123)
    full = torch.randn(1, total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    full_cu = torch.tensor([0, total // 3, total], dtype=torch.int32, device="cuda")

    x_ref = full.clone().requires_grad_()
    ref_out = core.out_proj(core(x_ref, full_cu, None))
    ref_out.sum().backward()

    local = total // world_size
    x_local = full[:, rank * local : (rank + 1) * local].transpose(0, 1).contiguous().requires_grad_()
    cp_out, _ = layer(x_local, packed_seq_params=packed(full_cu))
    loss = cp_out.sum()
    dist.all_reduce(loss)
    loss.backward()

    def gather_seq(t):
        parts = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=0)

    return rel_err(ref_out.detach()[0], gather_seq(cp_out.detach())[:, 0]), rel_err(
        x_ref.grad[0], gather_seq(x_local.grad)[:, 0]
    )


def main():
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=world_size)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        context_parallel_size=world_size,
    )
    torch.manual_seed(42)
    refs = {
        "gdn": ReplicatedGDN("qwen3_5", HIDDEN, DeltaRuleHeads(2, 4, 64, 64), torch.bfloat16).cuda(),
        "kda": ReplicatedKDA(HIDDEN, DeltaRuleHeads(4, 4, 64, 64), torch.bfloat16).cuda(),
    }
    failed = False
    try:
        for name, ref in refs.items():
            out_diff, grad_diff = check(ref, config, rank, world_size)
            ok = out_diff < 1e-2 and grad_diff < 1e-2
            failed |= not ok
            if rank == 0:
                print(
                    f"[{name} CP={world_size}] rel err fwd {out_diff:.2e}  dx {grad_diff:.2e}  {'PASS' if ok else 'FAIL'}"
                )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
