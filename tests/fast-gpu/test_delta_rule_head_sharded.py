"""Head-sharded GDN (Qwen3.5, Qwen3-Next layouts) and KDA against their replicated references at
TP = world size: forward output and every input / parameter gradient, gathered to the full layout.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_delta_rule_head_sharded.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads

sys.path.insert(0, os.path.dirname(__file__))
from delta_rule_reference import (  # noqa: E402
    ReplicatedGDN,
    ReplicatedKDA,
    build_layer,
    conv_of,
    gather,
    packed,
    rel_err,
    sharded_projections,
)

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

HIDDEN = 256
CASES = {
    "qwen3_5": DeltaRuleHeads(num_k_heads=4, num_v_heads=8, head_k_dim=64, head_v_dim=64),
    "qwen3_next": DeltaRuleHeads(num_k_heads=4, num_v_heads=8, head_k_dim=64, head_v_dim=64),
    "kda": DeltaRuleHeads(num_k_heads=8, num_v_heads=8, head_k_dim=64, head_v_dim=64),
}


def check(name, config, tp_group):
    torch.manual_seed(7)
    heads = CASES[name]
    ref = (
        ReplicatedKDA(HIDDEN, heads, torch.bfloat16)
        if name == "kda"
        else ReplicatedGDN(name, HIDDEN, heads, torch.bfloat16)
    ).cuda()
    layer = build_layer(ref, config)
    core = layer.linear_attn

    torch.manual_seed(11)
    x = torch.randn(1, 512, HIDDEN, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 200, 512], device="cuda", dtype=torch.int32)
    x_ref, x_new = x.clone().requires_grad_(), x.clone().transpose(0, 1).contiguous().requires_grad_()
    out_ref = ref(x_ref, cu_seqlens)
    out_new, _ = layer(x_new, packed_seq_params=packed(cu_seqlens))
    torch.manual_seed(13)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_new.backward(grad_out.transpose(0, 1))

    norm_ref = ref.o_norm if name == "kda" else ref.norm
    out_proj_ref = ref.o_proj if name == "kda" else ref.out_proj
    errors = {
        "out": rel_err(out_ref, out_new.transpose(0, 1)),
        "dx": rel_err(x_ref.grad, x_new.grad.transpose(0, 1)),
        "conv1d": rel_err(conv_of(ref, grad=True), gather(core.conv1d.weight.grad, 0, tp_group)),
        "A_log": rel_err(ref.A_log.grad, gather(core.A_log.grad, 0, tp_group)),
        "dt_bias": rel_err(ref.dt_bias.grad, gather(core.dt_bias.grad, 0, tp_group)),
        "norm": rel_err(norm_ref.weight.grad, core.norm.weight.grad),
        "out_proj": rel_err(out_proj_ref.weight.grad, gather(core.out_proj.weight.grad, 1, tp_group)),
    }
    for proj, full_grad in sharded_projections(ref, grad=True).items():
        errors[proj] = rel_err(full_grad, gather(getattr(core, proj).weight.grad, 0, tp_group))
    if name == "kda":
        errors["f_a_proj"] = rel_err(ref.f_a_proj.weight.grad, core.f_a_proj.weight.grad)
    return errors


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    failures = {}
    for name in CASES:
        errors = check(name, config, tp_group)
        if rank == 0:
            print(f"[{name} TP={world}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
        bad = {k: v for k, v in errors.items() if v > (1e-6 if world == 1 and k == "out" else 2e-2)}
        if bad:
            failures[name] = bad
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    if rank == 0:
        print(f"TP={world} head-sharded delta-rule layers match their replicated references")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
