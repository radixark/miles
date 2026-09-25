"""Head-sharded GDN core under TP x CP vs a replicated single-rank reference.

Run with::

    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen_gdn_tp_cp_parity.py --tp 2 --cp 1 --hf-layout qwen3_5
    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen_gdn_tp_cp_parity.py --tp 1 --cp 2
    torchrun --nproc_per_node=4 tests/e2e/precision/test_qwen_gdn_tp_cp_parity.py --tp 2 --cp 2

Loads one HF ``linear_attn`` state dict into (a) a replicated ``GatedDeltaRuleAttentionCore``
(``mp_config=None``, ``fla`` backend) on every rank and (b) the TP-sharded core over Megatron's TP
group (``--backend``, default ``loom``), optionally under fla CP state passing, and compares the
output, the input gradient and every parameter gradient (TP-merged back to HF names).  Invoked as
``python3 file.py`` it self-bootstraps under torchrun with 4 processes and runs the three
configurations.
"""

import argparse
import os
import sys

import torch

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-c-8-gpu-b200", labels=["precision"], hardware=["blackwell"])


def _hf_state(layout, gen):
    def rn(*shape, s=0.02):
        return torch.randn(*shape, generator=gen) * s

    hf = {
        "A_log": torch.log(torch.rand(layout.num_v_heads, generator=gen) * 8 + 0.5),
        "dt_bias": rn(layout.num_v_heads, s=0.5) + 0.5,
        "conv1d.weight": rn(2 * layout.key_dim + layout.value_dim, 1, layout.conv_kernel_size, s=0.3),
        "norm.weight": 1.0 + rn(layout.head_v_dim, s=0.1),
        "out_proj.weight": rn(layout.hidden_size, layout.value_dim),
    }
    if layout.hf_layout == "qwen3_next":
        hf["in_proj_qkvz.weight"] = rn(
            layout.num_k_heads * (2 * layout.head_k_dim + 2 * layout.group * layout.head_v_dim), layout.hidden_size
        )
        hf["in_proj_ba.weight"] = rn(layout.num_k_heads * 2 * layout.group, layout.hidden_size, s=0.05)
    else:
        hf["in_proj_qkv.weight"] = rn(2 * layout.key_dim + layout.value_dim, layout.hidden_size)
        hf["in_proj_z.weight"] = rn(layout.value_dim, layout.hidden_size)
        hf["in_proj_b.weight"] = rn(layout.num_v_heads, layout.hidden_size, s=0.05)
        hf["in_proj_a.weight"] = rn(layout.num_v_heads, layout.hidden_size, s=0.05)
    return {name: value.cuda() for name, value in hf.items()}


def _load(core, layout, hf, tp_rank, tp_size):
    from miles_plugins.models.gdn_attention import hf_linear_attn_to_local

    local = hf_linear_attn_to_local(layout, hf, tp_rank=tp_rank, tp_size=tp_size)
    with torch.no_grad():
        for name, param in _linear_attn_params(core).items():
            param.copy_(local[name].to(dtype=param.dtype, device=param.device))


def _linear_attn_params(core):
    return {name: param for name, param in core.named_parameters() if not name.endswith("_extra_state")}


def run_case(tp: int, cp: int, backend: str, hf_layout: str, seq_lens: list[int], tol: float) -> bool:
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.gdn_attention import GatedDeltaRuleAttentionCore, GdnLayout, local_to_hf_linear_attn

    rank = dist.get_rank()
    ps.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(123)
    tp_group, cp_group = ps.get_tensor_model_parallel_group(), ps.get_context_parallel_group()
    tp_rank, cp_rank = ps.get_tensor_model_parallel_rank(), ps.get_context_parallel_rank()

    layout = GdnLayout(
        hidden_size=512,
        num_k_heads=4,
        num_v_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
        hf_layout=hf_layout,
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=layout.hidden_size,
        num_attention_heads=8,
        tensor_model_parallel_size=tp,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        perform_initialization=True,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
    )
    hf = _hf_state(layout, torch.Generator(device="cpu").manual_seed(1234))
    # bf16 parameters throughout (Megatron's Float16Module does this in training); A_log included on both sides
    sharded = (
        GatedDeltaRuleAttentionCore(
            layout, gdn_backend=backend, mp_config=config, tp_group=tp_group, params_dtype=torch.bfloat16
        )
        .cuda()
        .to(torch.bfloat16)
    )
    _load(sharded, layout, hf, tp_rank, tp)
    reference = (
        GatedDeltaRuleAttentionCore(
            layout, gdn_backend="fla", mp_config=None, tp_group=None, params_dtype=torch.bfloat16
        )
        .cuda()
        .to(torch.bfloat16)
    )
    _load(reference, layout, hf, 0, 1)

    total = sum(seq_lens)
    cu_global = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    gen_x = torch.Generator(device="cpu").manual_seed(77)
    x_full = torch.randn(1, total, layout.hidden_size, generator=gen_x).cuda().to(torch.bfloat16)
    dy_full = torch.randn(1, total, layout.hidden_size, generator=gen_x).cuda().to(torch.bfloat16)

    x_ref = x_full.clone().requires_grad_(True)
    y_ref = reference(x_ref, cu_seqlens=cu_global)
    y_ref.backward(dy_full)
    ref_grads = local_to_hf_linear_attn(
        layout, [{name: p.grad.detach().float() for name, p in _linear_attn_params(reference).items()}]
    )

    if cp > 1:
        from miles_plugins.models.cp_utils import build_gdn_cp_context

        sharded.cp_group, sharded.cp_rank, sharded.cp_world_size = cp_group, cp_rank, cp
        part = total // cp
        window = slice(cp_rank * part, (cp_rank + 1) * part)
        x_loc = x_full[:, window].clone().requires_grad_(True)
        dy_loc = dy_full[:, window]
        cp_context = build_gdn_cp_context(sharded, cu_global, x_loc.device)
        y = sharded(x_loc, cu_seqlens=cp_context.cu_seqlens, cp_context=cp_context)
        y_ref_loc, dx_ref_loc = y_ref[:, window], x_ref.grad[:, window]
    else:
        x_loc = x_full.clone().requires_grad_(True)
        dy_loc = dy_full
        y = sharded(x_loc, cu_seqlens=cu_global)
        y_ref_loc, dx_ref_loc = y_ref, x_ref.grad
    y.backward(dy_loc)

    if backend == "loom":
        # bit determinism of the deterministic path: a second identical pass reproduces output and gradients
        x_again = x_loc.detach().clone().requires_grad_(True)
        y_again = (
            sharded(x_again, cu_seqlens=cu_global)
            if cp == 1
            else sharded(x_again, cu_seqlens=cp_context.cu_seqlens, cp_context=cp_context)
        )
        assert torch.equal(y_again, y), "loom forward is not bit-deterministic"
        grads_before = {name: p.grad.detach().clone() for name, p in _linear_attn_params(sharded).items()}
        for p in _linear_attn_params(sharded).values():
            p.grad = None
        y_again.backward(dy_loc)
        assert torch.equal(x_again.grad, x_loc.grad), "loom input gradient is not bit-deterministic"
        for name, p in _linear_attn_params(sharded).items():
            assert torch.equal(p.grad, grads_before[name]), f"loom gradient of {name} is not bit-deterministic"

    local_grads = {name: p.grad.detach().float().clone() for name, p in _linear_attn_params(sharded).items()}
    if cp > 1:
        for tensor in local_grads.values():
            dist.all_reduce(tensor, group=cp_group)
    gathered: list = [None] * tp
    dist.all_gather_object(gathered, {name: value.cpu() for name, value in local_grads.items()}, group=tp_group)
    gathered = [{name: value.cuda() for name, value in shard.items()} for shard in gathered]
    shard_grads = local_to_hf_linear_attn(layout, gathered)
    # the replicated norm weight already carries the TP-summed gradient on every rank

    def check(name, actual, expected):
        actual = actual.float()
        expected = expected.float()
        err = (actual - expected).abs().max().item()
        scale = expected.abs().max().item() + 1e-12
        ok = torch.allclose(actual, expected, atol=tol * scale, rtol=tol)
        return ok, f"{name:24s} maxabs={err:.3e} refmax={scale:.3e} rel={err / scale:.3e} {'OK' if ok else 'FAIL'}"

    checks = [("output", y, y_ref_loc), ("dhidden", x_loc.grad, dx_ref_loc)]
    checks += [(f"grad {name}", shard_grads[name], ref_grads[name]) for name in sorted(ref_grads)]
    ok_all = True
    lines = []
    for name, actual, expected in checks:
        ok, line = check(name, actual, expected)
        ok_all &= ok
        lines.append(line)
    failed = torch.tensor([0 if ok_all else 1], device="cuda")
    dist.all_reduce(failed)
    if rank == 0 or not ok_all:
        print(
            f"[rank {rank} tp_rank {tp_rank} cp_rank {cp_rank}] TP={tp} CP={cp} backend={backend} layout={hf_layout} seq_lens={seq_lens}"
        )
        print("\n".join("  " + line for line in lines))
        print("  RESULT:", "PASS" if failed.item() == 0 else "FAIL", flush=True)
    dist.barrier()
    ps.destroy_model_parallel()
    return failed.item() == 0


def main() -> int:
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--cp", type=int, default=None)
    parser.add_argument("--backend", default="loom")
    parser.add_argument("--hf-layout", default="qwen3_next")
    parser.add_argument("--seq-lens", default="300,724")
    parser.add_argument("--tol", type=float, default=2.5e-2)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    if args.tp is not None:
        cases = [(args.tp, args.cp or 1, args.hf_layout, seq_lens)]
    else:
        world = dist.get_world_size()
        cases = [(2, 1, "qwen3_5", seq_lens), (2, 1, "qwen3_next", seq_lens), (1, 2, "qwen3_next", seq_lens)]
        if world >= 4:
            cases.append((2, 2, "qwen3_next", [500, 1548]))
        cases = [(tp, cp, layout, lens) for tp, cp, layout, lens in cases if tp * cp <= world]
    ok = True
    for tp, cp, layout, lens in cases:
        ok &= run_case(tp, cp, args.backend, layout, lens, args.tol)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__, *sys.argv[1:]])
    raise SystemExit(main())
