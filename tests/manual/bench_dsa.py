"""DSA sparse attention fwd / fwd+bwd speed and precision at the attention configs of the latest models,
per TP-local head count and sequence length. One GPU.

    python tests/manual/bench_dsa.py [seq_len ...]

Backends: ``tilelang`` and ``flash_mla`` forwards (both with the TileLang backward). Precision is the
relative error against an fp32 dense reference (seq_len <= 8k) and between the two backends.
"""

import importlib.util
import pathlib
import sys

import torch

from miles.kernels.attention.dsa import sparse_attention
from miles.kernels.attention.dsa.sparse_attention import flash_mla_sparse_fwd

_spec = importlib.util.spec_from_file_location(
    "dsa_reference", pathlib.Path(__file__).parents[1] / "fast-gpu/kernels/attention/dsa/dsa_reference.py"
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

# name: (total heads, d_qk, d_v, topk, sink, S_kv / S, TP sizes used in training)
MODELS = {
    "glm5.3-flash (as in #2786: 64-wide zero tail)": (64, 576, 512, 2112, False, 1.0, (8, 4, 1)),
    "glm5.3-flash (no zero tail)": (64, 512, 512, 2112, False, 1.0, (8, 4, 1)),
    "dsv4.1-flash ratio-1 layers": (64, 512, 512, 640, True, 2.0, (8, 2, 1)),
    "dsv4.1-flash ratio-0 layers": (64, 512, 512, 640, True, 1.0, (8,)),
    "glm5": (64, 576, 512, 2048, False, 1.0, (8, 4, 1)),
}


def cuda_ms(fn, iters):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def causal_indices(seq_len, seq_len_kv, topk):
    rows = []
    for chunk in torch.arange(seq_len, device="cuda").split(4096):
        scores = torch.rand(len(chunk), seq_len_kv, device="cuda")
        limit = (chunk.view(-1, 1) + 1) * seq_len_kv // seq_len
        scores = scores.masked_fill(torch.arange(seq_len_kv, device="cuda").view(1, -1) >= limit, -1)
        idx = scores.topk(topk, dim=-1).indices
        rows.append(idx.masked_fill(torch.gather(scores, 1, idx) < 0, -1).int())
    return torch.cat(rows).view(1, seq_len, 1, topk)


def main():
    seq_lens = [int(s) for s in sys.argv[1:]] or [4096, 8192, 16384, 32768, 65536, 131072]
    backends = ["tilelang"] + (["flash_mla"] if flash_mla_sparse_fwd is not None else [])
    for name, (total_heads, d_qk, d_v, topk, sink, kv_ratio, tps) in MODELS.items():
        for tp in tps:
            heads = total_heads // tp
            for seq_len in seq_lens:
                torch.manual_seed(0)
                seq_len_kv = int(seq_len * kv_ratio)
                row = f"[{name:44s} TP={tp} heads={heads:3d} seq={seq_len:6d}]"
                try:
                    q = torch.randn(1, seq_len, heads, d_qk, device="cuda", dtype=torch.bfloat16)
                    kv = torch.randn(1, seq_len_kv, 1, d_qk, device="cuda", dtype=torch.bfloat16)
                    idx = causal_indices(seq_len, seq_len_kv, topk)
                    attn_sink = torch.randn(heads, device="cuda") if sink else None
                    scale = d_qk**-0.5
                    iters = max(2, min(10, 131072 // seq_len))
                    outs = {}
                    for backend in backends:
                        qg, kvg = q.clone().requires_grad_(), kv.clone().requires_grad_()
                        sg = attn_sink.clone().requires_grad_() if sink else None

                        def fwd(qg=qg, kvg=kvg, sg=sg, b=backend, idx=idx, scale=scale, d_v=d_v):
                            return sparse_attention(qg, kvg, idx, scale, d_v=d_v, attn_sink=sg, forward_backend=b)

                        with torch.no_grad():
                            t_fwd = cuda_ms(fwd, iters)
                        t_all = cuda_ms(lambda fwd=fwd: fwd().float().sum().backward(), iters)
                        qg.grad = kvg.grad = None
                        out = fwd()
                        out.float().sum().backward()
                        outs[backend] = (out.detach(), qg.grad, kvg.grad)
                        row += f"  {backend} fwd {t_fwd:8.2f} fwd+bwd {t_all:8.2f} ms"
                    if len(outs) == 2:
                        a, b = outs["tilelang"], outs["flash_mla"]
                        row += f"  backends agree out {reference.rel_diff(a[0], b[0]):.0e} dq {reference.rel_diff(a[1], b[1]):.0e} dkv {reference.rel_diff(a[2], b[2]):.0e}"
                    if seq_len <= 8192:
                        ref = reference.sparse_attention_ref(q, kv, idx, scale, d_v, attn_sink)
                        row += "  vs fp32 " + " ".join(
                            f"{k} {reference.rel_diff(ref, v[0]):.0e}" for k, v in outs.items()
                        )
                except torch.cuda.OutOfMemoryError:
                    row += "  OOM"
                torch.cuda.empty_cache()
                print(row, flush=True)


if __name__ == "__main__":
    main()
