"""Per-kernel device time of the DSA backends at production shapes: ``loom`` (generated) vs ``tilelang``.

Both kernel families run in this one process (the generated kernels are torch CUDA extensions), so the
same inputs and the same timing method apply to both.  Timing = CUDA kernel device time from
``torch.profiler`` (CUPTI activity records) summed per operator call; an explicit 256 MiB write flushes L2
before every timed call and is excluded from the sum.  Reported as the median over ``--iters`` calls.

    python tests/manual/models/dsa_train/bench_dsa_backends.py --json results.json
    python tests/manual/models/dsa_train/bench_dsa_backends.py --only glm52_tp4_attention,dsv4_tp1_attention

Shapes (see ``--list``): the GLM-5.2 5-layer e2e (TP4: 16 local heads, index heads from the checkpoint,
topk 2048, d_qk 576) and the DeepSeek-V4-Flash 4-layer e2e (TP1: 64 heads, d 512, window 128 + 512
compressed keys, indexer 64 heads over S/4 compressed keys), at 2048 and 4096 query rows per rank.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass

import torch
from torch.profiler import ProfilerActivity, profile

FLUSH_BYTES = 256 << 20
FLUSH_KERNEL_MARKERS = ("fill", "Fill", "elementwise_kernel")  # the L2-flush write; excluded from sums


@dataclass(frozen=True)
class AttentionShape:
    label: str
    layout: str  # thd (GLM-5) or bshd (DeepSeek-V4)
    batch: int
    rows: int
    heads: int
    d_tail: int
    topk: int
    sink: bool


@dataclass(frozen=True)
class IndexerShape:
    label: str
    layout: str  # thd (GLM-5) or sbhd (DeepSeek-V4)
    batch: int
    rows: int
    heads: int
    keys: int
    topk: int


def production_shapes(rows: int, glm_index_heads: int):
    attention = [
        AttentionShape(f"glm52_tp4_attention_r{rows}", "thd", 1, rows, 16, 64, 2048, False),
        AttentionShape(f"glm52_tp1_attention_r{rows}", "thd", 1, rows, 64, 64, 2048, False),
        AttentionShape(f"dsv4_tp1_attention_r{rows}", "bshd", 1, rows, 64, 0, 128 + 512, True),
    ]
    indexer = [
        IndexerShape(f"glm52_indexer_r{rows}", "thd", 1, rows, glm_index_heads, rows, 2048),
        IndexerShape(f"dsv4_indexer_r{rows}", "sbhd", 1, rows, 64, rows // 4, 512),
    ]
    return attention, indexer


def _causal_indices(rows, topk, keys, device, gen, offset=0):
    idx = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    for r in range(rows):
        limit = max(1, min(keys, (r + 1) * keys // rows))
        count = min(topk, limit)
        idx[r, :count] = torch.randperm(limit, generator=gen, device=device)[:count].int() + offset
    return idx


def make_attention_inputs(shape: AttentionShape, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(0)
    d_qk = 512 + shape.d_tail
    if shape.layout == "thd":
        q = torch.randn(shape.rows, shape.heads, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        kv = torch.randn(shape.rows, 1, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        indices = _causal_indices(shape.rows, shape.topk, shape.rows, device, gen).unsqueeze(1)
        do = torch.randn(shape.rows, shape.heads, 512, device=device, dtype=torch.bfloat16, generator=gen)
    else:
        q = torch.randn(shape.batch, shape.rows, shape.heads, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        kv = torch.randn(shape.batch, shape.rows, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        indices = torch.stack([_causal_indices(shape.rows, shape.topk, shape.rows, device, gen) for _ in range(shape.batch)])
        do = torch.randn(shape.batch, shape.rows, shape.heads, 512, device=device, dtype=torch.bfloat16, generator=gen)
    sink = torch.randn(shape.heads, device=device, generator=gen) if shape.sink else None
    return {"q": q, "kv": kv, "indices": indices, "do": do, "sink": sink, "sm_scale": 1.0 / math.sqrt(d_qk)}


def make_indexer_inputs(shape: IndexerShape, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(0)
    if shape.layout == "thd":
        q = torch.randn(shape.rows, shape.heads, 128, device=device, dtype=torch.bfloat16, generator=gen)
        k = torch.randn(shape.keys, 128, device=device, dtype=torch.bfloat16, generator=gen)
        w = torch.randn(shape.rows, shape.heads, device=device, generator=gen)
        ks = torch.zeros(shape.rows, device=device, dtype=torch.int32)
        ke = (torch.arange(shape.rows, device=device) + 1).clamp(max=shape.keys).int()
        topk_indices = _causal_indices(shape.rows, shape.topk, shape.keys, device, gen)
    else:
        q = torch.randn(shape.rows, shape.batch, shape.heads, 128, device=device, dtype=torch.bfloat16, generator=gen)
        k = torch.randn(shape.keys, shape.batch, 128, device=device, dtype=torch.bfloat16, generator=gen)
        w = torch.randn(shape.rows, shape.batch, shape.heads, device=device, generator=gen)
        ks = torch.zeros(shape.rows, device=device, dtype=torch.int32)
        ke = ((torch.arange(shape.rows, device=device) + 1) // (shape.rows // shape.keys)).int()
        topk_indices = None
    grad_scores = torch.randn(shape.rows, shape.topk, device=device, generator=gen) if topk_indices is not None else None
    return {"q": q, "k": k, "w": w, "ks": ks, "ke": ke, "topk_indices": topk_indices, "grad_scores": grad_scores}


class DeviceTimer:
    """Median CUDA device time (ms) of ``fn`` over ``iters`` cold-L2 calls, from CUPTI kernel records."""

    def __init__(self, warmup: int, iters: int, device="cuda"):
        self.warmup, self.iters = warmup, iters
        self.flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device)

    def __call__(self, fn) -> float:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(self.iters):
            self.flush.fill_(1)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            total_us = 0.0
            for event in prof.events():
                if event.device_type.name != "CUDA":
                    continue
                total_us += event.device_time if hasattr(event, "device_time") else event.cuda_time
            samples.append(total_us / 1000.0)
        return statistics.median(samples)


def bench_attention(shape, timer, backends):
    from miles_plugins.models import dsa_train

    inputs = make_attention_inputs(shape)
    q, kv, idx, do, sink, sm_scale = (inputs[k] for k in ("q", "kv", "indices", "do", "sink", "sm_scale"))
    row = asdict(shape)

    def loom_fwd():
        return dsa_train.sparse_attention_forward(q, kv, idx, sm_scale=sm_scale, attn_sink=sink, layout=shape.layout)

    o, lse = loom_fwd()
    row["loom_fwd_ms"] = timer(loom_fwd)
    row["loom_bwd_ms"] = timer(
        lambda: dsa_train.sparse_attention_backward(q, kv, o, do, idx, lse, sm_scale=sm_scale, attn_sink=sink, layout=shape.layout)
    )
    if "tilelang" in backends:
        if shape.layout == "thd":
            from miles_plugins.models.glm5.ops.tilelang_sparse_mla_bwd import sparse_mla_bwd
            from miles_plugins.models.glm5.ops.tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

            idx_c, q_c, kv_c = idx.contiguous(), q.contiguous(), kv.contiguous()
            tl_o, tl_lse = sparse_mla_fwd_interface(q_c, kv_c, idx_c, sm_scale=sm_scale)
            row["tilelang_fwd_ms"] = timer(lambda: sparse_mla_fwd_interface(q_c, kv_c, idx_c, sm_scale=sm_scale))
            row["tilelang_bwd_ms"] = timer(lambda: sparse_mla_bwd(q_c, kv_c, tl_o, do.contiguous(), idx_c, tl_lse, sm_scale=sm_scale))
        else:
            from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd import sparse_mqa_bwd_interface
            from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

            tl_o, tl_lse = sparse_mqa_fwd_interface(q, kv, sink, idx, sm_scale=sm_scale)
            row["tilelang_fwd_ms"] = timer(lambda: sparse_mqa_fwd_interface(q, kv, sink, idx, sm_scale=sm_scale))
            row["tilelang_bwd_ms"] = timer(lambda: sparse_mqa_bwd_interface(q, kv, sink, tl_o, do, idx, tl_lse, sm_scale=sm_scale))
    return row


def bench_indexer(shape, timer, backends):
    from miles_plugins.models import dsa_train

    inputs = make_indexer_inputs(shape)
    q, k, w, ks, ke = (inputs[n] for n in ("q", "k", "w", "ks", "ke"))
    row = asdict(shape)
    row["loom_logits_ms"] = timer(lambda: dsa_train.indexer_logits(q, k, w, ks, ke, layout=shape.layout))
    if shape.layout == "thd":
        topk_indices, grad_scores = inputs["topk_indices"], inputs["grad_scores"]
        row["loom_bwd_ms"] = timer(lambda: dsa_train.indexer_backward(q, k, w, topk_indices, grad_scores, layout="thd"))
    if "tilelang" in backends:
        if shape.layout == "thd":
            from miles_plugins.models.glm5.ops.tilelang_indexer_bwd import indexer_bwd_interface
            from miles_plugins.models.glm5.ops.tilelang_indexer_fwd import indexer_fwd_interface

            row["tilelang_logits_ms"] = timer(lambda: indexer_fwd_interface(q, k, w, ks, ke, clean_logits=True))
            row["tilelang_bwd_ms"] = timer(lambda: indexer_bwd_interface(q, w, k, topk_indices, grad_scores))
        else:
            from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd import batched_indexer_fwd

            row["tilelang_logits_ms"] = timer(lambda: batched_indexer_fwd(q, k, w, ks, ke))
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 4096], help="query rows per rank")
    parser.add_argument("--glm-index-heads", type=int, default=64, help="GLM-5.2 index_n_heads (checkpoint config)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backends", default="loom,tilelang")
    parser.add_argument("--only", default="", help="comma-separated label prefixes")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    backends = set(args.backends.split(","))
    if "tilelang" in backends:
        try:
            import tilelang  # noqa: F401
        except ImportError:
            print("tilelang not importable: timing the loom backend only", file=sys.stderr)
            backends.discard("tilelang")
    only = [p for p in args.only.split(",") if p]
    selected = []
    for rows in args.rows:
        attention, indexer = production_shapes(rows, args.glm_index_heads)
        selected += [(bench_attention, s) for s in attention] + [(bench_indexer, s) for s in indexer]
    if only:
        selected = [(f, s) for f, s in selected if any(s.label.startswith(p) for p in only)]
    if args.list:
        for _, s in selected:
            print(s.label, asdict(s))
        return 0
    timer = DeviceTimer(args.warmup, args.iters)
    rows = []
    for fn, shape in selected:
        rows.append(fn(shape, timer, backends))
        print(json.dumps(rows[-1]), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": torch.cuda.get_device_name(), "backends": sorted(backends), "rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
