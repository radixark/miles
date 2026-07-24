# ruff: noqa
"""Performance benchmark for the GLM-5 DSA sparse-MLA backward kernels.

Sweeps the query sequence length S over {2048, 4096, 8192, 16384} at the fixed
GLM-5 DSA latent geometry (DQKV=576 = kv_lora_rank 512 + qk_rope 64, DV=512) and
the model's real ``index_topk=2048``, comparing the two TileLang backward
implementations that share an identical public API and math:

  * ``tilelang_sparse_mla_bwd``     -- the vendored reference kernel.
  * ``tilelang_sparse_mla_bwd_opt`` -- the pipelined split-D + manually swizzle + GEMM-operand-reuse variant 
  the production ``sparse_mla.py`` autograd path uses (parity cosine ~1.0 vs the reference).

Timing: TileLang ``do_bench`` (JIT-warmed, L2 flushed per iter so gathered-KV isn't served
warm; mean wall-clock ms). ms -> TFLOP/s / TB/s uses the DeepSeek-V3.2 accounting: 5 GEMMs
per selected KV per token, full ``topk`` work per token (masked entries still flow through
the GEMMs). Inputs are synthetic (bf16 q/kv/dO, causal top-k indices, plausible LSE, real
``preprocess`` Delta); the backward is value-independent and both kernels compute the same
math, so parity + timing hold for any inputs and run unmodified at S=16384 / arbitrary heads.

Full ``sparse_mla_bwd`` (pre+bwd+post) is timed by default; ``--kernel-only`` isolates the
``bwd`` kernel (the sole part that differs).

TP -> threads: GLM-5/5.1/5.2 share ``num_attention_heads=64``, ``kv_group=1``; Megatron
shards the MLA query heads over attention-TP (KV latent replicated), so the kernel sees
``H = 64 // TP`` and ``tilelang_sparse_mla_bwd_opt``'s adaptive gate picks threads off it
(``--heads`` stands in for ``64 // TP``): 64 (TP1) -> 256; 32 (TP2) / 16 (TP4) -> 128
(8 warps can't tile M<64); 8 (TP8) is out-of-bounds (padded_H=16>8), so keep attention-TP <= 4.

Run (always use the pinned env):
        python miles_plugins/models/glm5/ops/benchmark_sparse_mla_bwd.py
        python miles_plugins/models/glm5/ops/benchmark_sparse_mla_bwd.py --kernel-only
        python miles_plugins/models/glm5/ops/benchmark_sparse_mla_bwd.py --heads 128 --seq-lens 4096,8192
"""

from __future__ import annotations

import argparse
import os
import sys

# TileLang's auto-target detection + nvcc lookup both need CUDA_HOME; the pixi
# ``nemo`` env doubles as the CUDA root, derived here from this interpreter.
os.environ.setdefault("CUDA_HOME", os.path.dirname(os.path.dirname(sys.executable)))

import torch

from miles_plugins.models.glm5.ops import tilelang_sparse_mla_bwd as ref_mod
from miles_plugins.models.glm5.ops import tilelang_sparse_mla_bwd_opt as opt_mod

# Fixed by the GLM-5 DSA / DeepSeek sparse-MLA latent geometry (the kernels hard-assert
# dim_plus_tail == 576 and D (value width) == 512).
DQKV = 576  # kv_lora_rank(512) + qk_rope_head_dim(64) -- the score/dQ/dKV width
DV = 512  # kv_lora_rank -- the dP/dV width

DEFAULT_SEQ_LENS = [2048, 4096, 8192, 16384]


def _causal_topk_indices(seq_len: int, seq_len_kv: int, topk: int, device, generator) -> torch.Tensor:
    """Causal top-k gather indices ``[S, topk]`` int32 (vectorized, fast at S=16384).

    Token ``t`` selects ``min(t+1, topk)`` random keys from ``[0, t]``; the tail stays
    ``-1`` (masked). Distinctness is not enforced (irrelevant to timing/parity); the
    ``-1`` sentinel and the causal key range mirror the real DSA index pattern.
    """
    t = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    col = torch.arange(topk, device=device).unsqueeze(0)  # [1, topk]
    n_valid = torch.clamp(t + 1, max=topk)  # [S, 1] valid count per row
    rand = torch.rand(seq_len, topk, device=device, generator=generator)
    keys = (rand * (t + 1).float()).floor().long().clamp_(0, seq_len_kv - 1)  # in [0, t]
    idx = torch.where(col < n_valid, keys, torch.full_like(keys, -1))
    return idx.to(torch.int32)


def _build_inputs(seq_len, seq_len_kv, heads, topk, kv_group, device, generator):
    """Synthetic un-batched backward inputs ``(q, kv, o, do, indices, lse)``.

    Layouts: ``q[S, H, 576]``, ``kv[S_kv, G, 576]``, ``o/do[S, H, 512]``,
    ``indices[S, G, topk]`` int32, ``lse[S, H]`` fp32.
    """
    q = torch.randn(seq_len, heads, DQKV, device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    kv = torch.randn(seq_len_kv, kv_group, DQKV, device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    o = torch.randn(seq_len, heads, DV, device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    do = torch.randn(seq_len, heads, DV, device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    indices = (
        _causal_topk_indices(seq_len, seq_len_kv, topk, device, generator).view(seq_len, kv_group, topk).contiguous()
    )
    # A plausible LSE (~log-sum-exp scale over the valid keys); parity/timing are
    # insensitive to its exact value, both kernels consume the identical tensor.
    lse = torch.randn(seq_len, heads, device=device, dtype=torch.float32, generator=generator) + 4.0
    return q, kv, o, do, indices, lse.contiguous()


def _flops_bandwidth(ms, seq_len, heads, topk):
    """Convert mean wall-clock ``ms`` to (TFLOP/s, IO TB/s) using the upstream accounting."""
    per_token_flop = 2 * (
        heads * DV * topk  # dP  : dO @ KV^T
        + heads * DQKV * topk  # P   : Q  @ KV^T (fwd score recompute)
        + heads * DQKV * topk  # dQ  : dP @ KV
        + heads * DQKV * topk  # dKV : dP^T @ Q
        + heads * DV * topk  # dV  : P^T @ dO
    )
    tflops = per_token_flop * seq_len / (ms * 1e-3) / 1e12
    io_tbs = seq_len * max(DQKV * 2, DQKV + DV) * topk * 2 / (ms * 1e-3) / 1e12
    return tflops, io_tbs


def _check_parity(inputs, sm_scale):
    """Return (dq_cos, dkv_cos) between the reference and fusion kernels' outputs."""
    q, kv, o, do, indices, lse = inputs
    dq_ref, dkv_ref = ref_mod.sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale)
    dq_opt, dkv_opt = opt_mod.sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale)
    cos = torch.nn.functional.cosine_similarity
    dq_cos = cos(dq_opt.float().flatten(), dq_ref.float().flatten(), dim=0).item()
    dkv_cos = cos(dkv_opt.float().flatten(), dkv_ref.float().flatten(), dim=0).item()
    return dq_cos, dkv_cos


def _bench_e2e(mod, inputs, sm_scale, warmup, rep, backend, threads):
    """Benchmark the full backward (preprocess + bwd + postprocess) at an explicit ``threads``.

    Builds the kernels directly (rather than via ``mod.sparse_mla_bwd``) so the same
    ``threads`` is applied to both variants -- same-condition comparison.
    """
    from tilelang.profiler import do_bench

    q, kv, o, do, indices, lse = inputs
    q4, kv4, o4, do4 = (t.unsqueeze(0) for t in (q, kv, o, do))
    indices4, lse4 = indices.unsqueeze(0), lse.unsqueeze(0)
    B, S, H, dim = q4.shape
    _, S_kv, kv_group, _ = kv4.shape
    D, D_tail, topk = DV, dim - DV, indices4.shape[-1]

    preprocess_kernel = mod.preprocess(B, S, H, D)
    bwd_kernel = mod.bwd(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, True, threads=threads)
    postprocess_kernel = mod.postprocess(B, S_kv, D, D_tail, kv_group)

    def fn():
        delta = preprocess_kernel(o4, do4)
        dkv = torch.zeros_like(kv4, dtype=torch.float32)
        dq = bwd_kernel(q4, kv4, do4, indices4, lse4, delta, dkv)
        return dq, postprocess_kernel(dkv)

    return do_bench(fn, warmup=warmup, rep=rep, backend=backend)


def _bench_kernel_only(mod, inputs, sm_scale, warmup, rep, backend, threads):
    """Benchmark only the ``bwd`` kernel at an explicit ``threads`` (the sole part that differs)."""
    from tilelang.profiler import do_bench

    q, kv, o, do, indices, lse = inputs
    q4, kv4, o4, do4 = (t.unsqueeze(0) for t in (q, kv, o, do))
    indices4, lse4 = indices.unsqueeze(0), lse.unsqueeze(0)
    B, S, H, dim = q4.shape
    _, S_kv, kv_group, _ = kv4.shape
    D, D_tail, topk = DV, dim - DV, indices4.shape[-1]

    preprocess_kernel = mod.preprocess(B, S, H, D)
    bwd_kernel = mod.bwd(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, True, threads=threads)
    delta = preprocess_kernel(o4, do4)
    dkv = torch.zeros_like(kv4, dtype=torch.float32)

    def fn():
        return bwd_kernel(q4, kv4, do4, indices4, lse4, delta, dkv)

    return do_bench(fn, warmup=warmup, rep=rep, backend=backend)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def _adaptive_threads(heads: int, kv_group: int) -> int:
    """Mirror the opt ``bwd`` gate: 256 threads once block_H>=64, else 128."""
    block_h = min(64, max(1 << (heads // kv_group - 1).bit_length(), 16))
    return 256 if block_h >= 64 else 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=",".join(map(str, DEFAULT_SEQ_LENS)),
        help="comma-separated query seq lengths S to sweep (default: 2048,4096,8192,16384)",
    )
    parser.add_argument("--heads", type=int, default=64, help="number of query heads H (default: 64)")
    parser.add_argument(
        "--topk", type=int, default=2048, help="selected KVs/token; multiple of 64 (GLM index_topk=2048)"
    )
    parser.add_argument("--kv-group", type=int, default=1, help="KV groups (default: 1)")
    parser.add_argument("--warmup", type=float, default=250.0, help="do_bench target warmup ms (default: 250)")
    parser.add_argument("--rep", type=float, default=100.0, help="do_bench target measure ms (default: 100)")
    parser.add_argument(
        "--backend",
        choices=["event", "cupti", "cudagraph"],
        default="event",
        help="do_bench timing backend (default: event; cupti excludes host overhead)",
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="time only the bwd kernel (the part that differs), Delta precomputed",
    )
    parser.add_argument("--no-check", action="store_true", help="skip the per-shape reference-vs-fusion parity check")
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed (default: 0)")
    parser.add_argument(
        "--threads",
        type=str,
        default="adaptive",
        help="threads for BOTH kernels (same-condition): 'adaptive' (256 if block_H>=64 else 128) or an int",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to build and run the TileLang sparse-MLA backward kernels.")
    if args.topk % 64 != 0:
        raise SystemExit(f"--topk must be a multiple of 64, got {args.topk}.")

    seq_lens = _parse_int_list(args.seq_lens)
    device = torch.device("cuda")
    sm_scale = DQKV**-0.5  # the kernels' own default (D + D_tail) ** -0.5; passed explicitly.
    threads = _adaptive_threads(args.heads, args.kv_group) if args.threads == "adaptive" else int(args.threads)

    print(f"device        : {torch.cuda.get_device_name(device)}")
    print(f"latent geom   : DQKV={DQKV} DV={DV}  dtype=bf16  sm_scale={sm_scale:.6f}  (GLM-5 DSA)")
    print(f"fixed shape   : B=1 H={args.heads} topk={args.topk} kv_group={args.kv_group}  (causal self-attn, S_kv=S)")
    print(f"seq-len sweep : {seq_lens}")
    print(f"threads       : {threads} ({args.threads}; SAME for ref and opt -> speedup isolates the fusion)")
    print(f"timing        : do_bench(warmup={args.warmup:g}ms, rep={args.rep:g}ms, backend={args.backend})")
    print(f"mode          : {'bwd kernel only' if args.kernel_only else 'full sparse_mla_bwd (pre+bwd+post)'}")
    print()

    bench = _bench_kernel_only if args.kernel_only else _bench_e2e
    hdr = f"{'S':>8s}{'ref ms':>12s}{'fusion ms':>12s}{'speedup':>10s}{'ref TF/s':>10s}{'fus TF/s':>10s}"
    if not args.no_check:
        hdr += f"{'dq_cos':>10s}{'dkv_cos':>10s}"
    print(hdr)
    print("-" * len(hdr))

    for S in seq_lens:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        inputs = _build_inputs(S, S, args.heads, args.topk, args.kv_group, device, gen)

        parity_cols = ""
        if not args.no_check:
            dq_cos, dkv_cos = _check_parity(inputs, sm_scale)
            flag = "" if (dq_cos > 0.999 and dkv_cos > 0.999) else " !"
            parity_cols = f"{dq_cos:>10.6f}{dkv_cos:>10.6f}{flag}"

        ref_ms = bench(ref_mod, inputs, sm_scale, args.warmup, args.rep, args.backend, threads)
        fusion_ms = bench(opt_mod, inputs, sm_scale, args.warmup, args.rep, args.backend, threads)
        ref_tf, _ = _flops_bandwidth(ref_ms, S, args.heads, args.topk)
        fus_tf, _ = _flops_bandwidth(fusion_ms, S, args.heads, args.topk)
        speedup = ref_ms / fusion_ms

        print(
            f"{S:>8d}{ref_ms:>12.3f}{fusion_ms:>12.3f}{speedup:>9.3f}x{ref_tf:>10.1f}{fus_tf:>10.1f}{parity_cols}",
            flush=True,
        )

    print("-" * len(hdr))
    print("speedup = ref_ms / fusion_ms  (same threads for both -> isolates the fusion/split-D win)")


if __name__ == "__main__":
    main()
