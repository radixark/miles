"""Forward+backward of the linear-attention short conv (SiLU, width 4, packed sequences) per backend: fla's
"triton", fla's "mix" (causal-conv1d's CUDA backward) and "miles" (miles.kernels.conv's sequence-aligned
backward where ``_seqtile_applies``, fla's path elsewhere).

    python tests/manual/bench_short_conv.py

Times the forward alone and forward + backward (input and weight gradients; backward is their difference)
through ``triton.testing.do_bench_cudagraph(rep=20)``, the median of 7 rounds that alternate between backends
(the method of fla-org/flash-linear-attention#1328). Before timing, every backend's output and gradients are
checked against an fp32 per-sequence reference. Shapes: Kimi-K3's q/k/v convs at TP 1/2/4 (12288 / 6144 / 3072
channels, fp32 weight) under several packings, and the packed shapes of #1328 (bf16 weight here, no bias).
"""

import random
import statistics

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from miles_plugins.models import linear_attn

WIDTH = 4
ROUNDS = 7


def mixed_lengths(tokens, seed=0):
    """RL-like packing: a few long rollouts and many short ones."""
    rng = random.Random(seed)
    lengths = [tokens * 3 // 8, tokens // 8, tokens // 16]
    while sum(lengths) < tokens:
        lengths.append(min(rng.randint(64, 2048), tokens - sum(lengths)))
    rng.shuffle(lengths)
    return lengths


def cu_seqlens(tokens, packing):
    if packing == "mixed":
        lengths = mixed_lengths(tokens)
    elif packing == "fla":  # #1328's boundaries, with a 1-token sequence
        return torch.tensor([0, 1, tokens // 4 + 3, tokens // 2, tokens], dtype=torch.int32, device="cuda")
    else:
        lengths = [tokens // int(packing)] * int(packing)
        lengths[-1] += tokens - sum(lengths)
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")


SHAPES = [  # (tokens, channels, packing: sequence count | "mixed" | "fla", weight dtype)
    *[(65536, 12288, p, torch.float32) for p in ("16", "128", "1024", "4096", "mixed")],
    (65536, 6144, "16", torch.float32),
    (65536, 3072, "16", torch.float32),
    (16384, 12288, "16", torch.float32),
    (4096, 12288, "16", torch.float32),
    *[(t, c, "fla", torch.bfloat16) for t, c in ((1024, 1024), (2048, 2048), (8192, 2048), (32768, 4096))],
]


def reference(x, weight, cu):
    """fp32 per-sequence conv + SiLU."""
    bounds = cu.tolist()
    xs = x[0].float()
    segments = [
        F.conv1d(F.pad(xs[a:b].t()[None], (WIDTH - 1, 0)), weight.float()[:, None], groups=xs.shape[1])[0].t()
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    return F.silu(torch.cat(segments))[None]


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run_shape(tokens, channels, packing, weight_dtype, backends):
    torch.manual_seed(0)
    x = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16).requires_grad_()
    weight = (torch.randn(channels, WIDTH, device="cuda") * 0.5).to(weight_dtype).requires_grad_()
    dy = torch.randn_like(x)
    cu = cu_seqlens(tokens, packing)

    def fwd_bwd(backend):
        y = linear_attn.causal_short_conv(x, weight, "silu", backend, cu_seqlens=cu)
        return y, torch.autograd.grad(y, (x, weight), dy)

    xr, wr = x.detach().float().requires_grad_(), weight.detach().float().requires_grad_()
    y_ref = reference(xr, wr, cu)
    ref = (y_ref, *torch.autograd.grad(y_ref, (xr, wr), dy.float()))
    errors = {}
    for b in backends:
        y, grads = fwd_bwd(b)
        errors[b] = max(rel_err(out, r) for out, r in zip((y, *grads), ref, strict=True))
    # A live autograd graph from these eager runs would make CUDA graph capture fail (stale stream reference).
    del y, grads, y_ref

    # Forward alone (no autograd) and forward + backward; backward = their difference.
    def fwd(backend):
        with torch.no_grad():
            return linear_attn.causal_short_conv(x, weight, "silu", backend, cu_seqlens=cu)

    samples = {(b, mode): [] for b in backends for mode in ("fwd", "fwdbwd")}
    for _ in range(ROUNDS):
        for b in backends:
            for mode, fn in (("fwd", fwd), ("fwdbwd", fwd_bwd)):
                samples[b, mode].append(do_bench_cudagraph(lambda fn=fn, b=b: fn(b), rep=20, return_mode="median"))
    return {key: statistics.median(s) for key, s in samples.items()}, errors


def main():
    backends = ["triton", "mix", "miles"] if linear_attn.conv_bwd_kernel_available() else ["triton", "mix"]
    print(f"{torch.cuda.get_device_name()}  ms, cudagraph median of {ROUNDS} alternating rounds: fwd+bwd (fwd / bwd)")
    print(
        f"{'tokens x channels':>18} {'packing':>8} {'weight':>8} {'miles path':>10} "
        + "".join(f"{b:>24}" for b in backends)
        + "  max rel err"
    )
    for tokens, channels, packing, weight_dtype in SHAPES:
        cu = cu_seqlens(tokens, packing)
        seqtile = linear_attn._seqtile_applies(
            x=torch.empty(1, tokens, channels, dtype=torch.bfloat16, device="meta"),
            weight=torch.empty(channels, WIDTH, dtype=weight_dtype, device="meta"),
            activation="silu",
            cu_seqlens=cu,
            cp_context=None,
        )
        times, errors = run_shape(tokens, channels, packing, weight_dtype, backends)
        cells = []
        for b in backends:
            total, forward = times[b, "fwdbwd"], times[b, "fwd"]
            cells.append(f"{f'{total:.3f} ({forward:.3f} / {total - forward:.3f})':>24}")
        print(
            f"{f'{tokens} x {channels}':>18} {packing:>8} {str(weight_dtype)[6:]:>8} {'seqtile' if seqtile else 'fla':>10} "
            + "".join(cells)
            + "  "
            + " ".join(f"{b}={errors[b]:.1e}" for b in backends),
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
