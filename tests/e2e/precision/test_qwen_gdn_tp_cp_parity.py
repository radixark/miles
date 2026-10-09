"""Head-sharded GDN with the deterministic ``loom`` backend under TP, CP and TP x CP against the replicated fla
reference of ``tests/fast-gpu/linear_attn_reference.py``: forward output, input gradient and every parameter
gradient (gathered across TP, summed across CP), plus bit determinism of the loom path (a second identical pass
reproduces the output and all gradients exactly).  Needs Blackwell (SM100a / SM103a) GPUs.

    torchrun --nproc_per_node=4 tests/e2e/precision/test_qwen_gdn_tp_cp_parity.py
    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen_gdn_tp_cp_parity.py --tp 2 --cp 1 --family qwen3_5

Invoked as ``python3 file.py`` it self-bootstraps under torchrun with 4 processes and runs TP=2, CP=2 and
TP=2 x CP=2 for both HF projection layouts.
"""

import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "fast-gpu"))
from linear_attn_reference import (  # noqa: E402
    ReplicatedGDN,
    build_layer,
    conv_of,
    gather,
    packed,
    rel_err,
    sharded_projections,
)

register_cuda_ci(est_time=600, suite="stage-c-8-gpu-b200", labels=["precision"], hardware=["blackwell"])

HIDDEN = 512
# The loom kernels take K = V = 128; two value heads per key head exercise the grouped-head path.
HEADS = LinearAttnHeads(num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128)
SEQLENS = [304, 720]  # multiples of 2 * cp for cp <= 4, so the zigzag chunks are exact
TOLERANCE = 2e-2
CASES = [(2, 1, "qwen3_5"), (2, 1, "qwen3_next"), (1, 2, "qwen3_next"), (2, 2, "qwen3_5"), (2, 2, "qwen3_next")]


def _config(tp: int, cp: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=False,
    )


def _zigzag(full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    """This CP rank's zigzag shard of a ``[T, ...]`` packed stream (Megatron's CP token order)."""
    parts = []
    for segment in full.split(SEQLENS):
        chunks = segment.chunk(2 * size)
        parts += [chunks[rank], chunks[2 * size - 1 - rank]]
    return torch.cat(parts)


def run_case(tp: int, cp: int, family: str, backend: str) -> dict[str, float]:
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = parallel_state.get_context_parallel_rank()

    torch.manual_seed(7)
    ref = ReplicatedGDN(family, HIDDEN, HEADS, torch.bfloat16).cuda()
    layer = build_layer(ref, _config(tp, cp), allgather_cp=False, backend=backend)
    core = layer.linear_attn

    torch.manual_seed(11)
    x = torch.randn(1, sum(SEQLENS), HIDDEN, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)

    x_ref = x.clone().requires_grad_()
    out_ref = ref(x_ref, cu_seqlens)
    out_ref.backward(grad)

    def local(t):  # [1, T, hidden] -> this CP rank's zigzag shard in the layer's [t, 1, hidden] layout
        return _zigzag(t[0], cp_rank, cp).unsqueeze(1).contiguous()

    def forward_backward():
        x_local = local(x).requires_grad_()
        out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
        out.backward(local(grad))
        grads = {name: p.grad.detach().clone() for name, p in core.named_parameters()}
        return out.detach(), x_local.grad, grads

    out, dx, grads = forward_backward()
    errors = {}
    if backend == "loom":
        for p in core.parameters():
            p.grad = None
        out_again, dx_again, grads_again = forward_backward()
        errors["nondeterministic_out"] = float(not torch.equal(out, out_again))
        errors["nondeterministic_dx"] = float(not torch.equal(dx, dx_again))
        for name in grads:  # one entry per parameter whose gradient the second pass did not reproduce bit for bit
            if not torch.equal(grads[name], grads_again[name]):
                gap = (grads[name].float() - grads_again[name].float()).abs().max().item()
                errors[f"nondeterministic_grad[{name}]"] = gap

    if cp > 1:  # each CP rank saw a token shard: the full parameter gradient is the sum over the CP group
        for g in grads.values():
            dist.all_reduce(g, group=cp_group)
    errors["out"] = rel_err(local(out_ref), out)
    errors["dx"] = rel_err(local(x_ref.grad), dx)
    errors["A_log"] = rel_err(ref.A_log.grad, gather(grads["A_log"], 0, tp_group))
    errors["dt_bias"] = rel_err(ref.dt_bias.grad, gather(grads["dt_bias"], 0, tp_group))
    errors["norm"] = rel_err(ref.norm.weight.grad, grads["norm.weight"])
    errors["out_proj"] = rel_err(ref.out_proj.weight.grad, gather(grads["out_proj.weight"], 1, tp_group))
    for conv, full_grad in conv_of(ref, grad=True).items():
        errors[conv] = rel_err(full_grad, gather(grads[f"{conv}.weight"], 0, tp_group))
    for proj, full_grad in sharded_projections(ref, grad=True).items():
        errors[proj] = rel_err(full_grad, gather(grads[f"{proj}.weight"], 0, tp_group))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--cp", type=int, default=None)
    parser.add_argument("--family", choices=["qwen3_5", "qwen3_next"], default=None)
    parser.add_argument("--backend", default="loom")
    args = parser.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    if args.backend == "loom":
        from miles_plugins.models.gdn_chunk_train import SUPPORTED_CAPABILITIES

        if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
            if rank == 0:
                print(f"SKIPPED: the loom GDN kernels need an SM100a / SM103a GPU, got {torch.cuda.get_device_name()}")
            dist.destroy_process_group()
            return 0
    if args.tp is not None:
        cases = [(args.tp, args.cp or 1, args.family or "qwen3_next")]
    else:
        cases = [(tp, cp, family) for tp, cp, family in CASES if tp * cp <= world]

    failures = {}
    for tp, cp, family in cases:
        errors = run_case(tp, cp, family, args.backend)
        if rank == 0:
            print(f"[{family} TP={tp} CP={cp} {args.backend}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
        bad = {k: v for k, v in errors.items() if v > (0 if k.startswith("nondeterministic") else TOLERANCE)}
        if bad:
            failures[(family, tp, cp)] = bad
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if failures:
        print(f"FAILED (rank {rank}): {failures}")
        return 1
    if rank == 0:
        print(f"{args.backend} head-sharded GDN matches the replicated fla reference under TP / CP (world {world})")
    return 0


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__, *sys.argv[1:]])
    raise SystemExit(main())
