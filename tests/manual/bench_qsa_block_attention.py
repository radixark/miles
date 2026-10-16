"""QSA block attention speed per TP-local head count, sequence length and packing. One GPU.

    python tests/manual/bench_qsa_block_attention.py [--selections a.pt[,b.pt]] [--baseline old.py] [--heads 12/1,...]

Selections: random whole blocks by default (each query draws its blocks independently, the worst case for a tile
that walks the union of consecutive queries' blocks), or ``--selections`` files holding ``{"blocks": int32 [T, K]}``
for one long sequence, as an indexer selected them (a prefix of such a sequence is itself a valid selection).
``--baseline`` loads another kernel file exposing ``qsa_block_sparse_attention`` (the tensor-core kernel this one
replaced: ``git show <ref>:miles/kernels/attention/qsa/block_sparse_attention.py > /tmp/old.py``) and times it on the
same inputs, with the block bitmap its caller built.

Columns: prep (index or bitmap), forward, forward + backward, in ms (median of ``--iters`` calls); step = prep +
forward + forward + backward (one forward for the log-probs, then a recomputed forward and the backward); peak memory
of one such step above the inputs, in GiB.
"""

import argparse
import importlib.util
import statistics

import torch

from miles.kernels.attention.qsa import build_qsa_block_index, qsa_block_attention

RATIO = 4
BUDGET = 512  # blocks per query: indexer_budget 2048 / compress_ratio 4


def cuda_ms(fn, iters):
    """Median GPU time of ``iters`` back-to-back calls after a warm-up, so one stalled call does not move it."""
    fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def random_blocks(length, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_blocks = -(-length // RATIO)
    rows = []
    for c0 in range(0, length, 4096):
        pos = torch.arange(c0, min(length, c0 + 4096), device="cuda")
        scores = torch.rand(pos.numel(), n_blocks, device="cuda", generator=g)
        scores = scores.masked_fill(
            torch.arange(n_blocks, device="cuda")[None, :] >= ((pos + 1) // RATIO)[:, None], -1
        )
        top = scores.topk(min(BUDGET, n_blocks), dim=1)
        rows.append(torch.where(top.values >= 0, top.indices, -1))
    blocks = torch.cat(rows).to(torch.int32)
    return torch.nn.functional.pad(blocks, (0, BUDGET - blocks.shape[1]), value=-1)


def pack(parts):
    """Per-sequence block ids -> packed ids and cu_seqlens."""
    rows, lens, first = [], [], 0
    for blocks in parts:
        rows.append(torch.where(blocks >= 0, blocks + first, -1))
        lens.append(blocks.shape[0])
        first += -(-blocks.shape[0] // RATIO)
    cu = torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)
    return torch.cat(rows), cu


def bitmap_form(blocks, cu):
    """The inputs the replaced kernel took: [T, num_blocks] uint8 flags plus the per-sequence block grid."""
    lens = (cu[1:] - cu[:-1]).long()
    blocks_per_seq = -(-lens // RATIO)
    seq = torch.repeat_interleave(torch.arange(lens.numel(), device="cuda"), lens)
    block_seq = torch.repeat_interleave(torch.arange(lens.numel(), device="cuda"), blocks_per_seq)
    first_block = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    pos = torch.arange(blocks.shape[0], device="cuda") - cu[:-1].long()[seq]
    block_first = (
        cu[:-1].long()[block_seq] + (torch.arange(block_seq.numel(), device="cuda") - first_block[block_seq]) * RATIO
    )
    block_last = torch.minimum(block_first + RATIO, cu[1:].long()[block_seq]) - 1
    sel = torch.zeros(blocks.shape[0], block_seq.numel(), dtype=torch.uint8, device="cuda")
    rows = torch.arange(blocks.shape[0], device="cuda")[:, None].expand_as(blocks)
    sel[rows[blocks >= 0], blocks[blocks >= 0].long()] = 1
    tail = (pos + 1) % RATIO != 0  # the causal tail's block, when the query's own block is not complete
    sel[torch.arange(blocks.shape[0], device="cuda")[tail], (first_block[seq] + pos // RATIO)[tail]] = 1
    tok_base = cu[:-1][seq].int()
    blk_base = first_block[seq].int()
    hi = torch.arange(blocks.shape[0], device="cuda", dtype=torch.int32)
    return sel, tok_base, hi, blk_base, tok_base, block_first.int(), block_last.int()


def measure(attend, prep, q, k, v, iters):
    gout = torch.randn_like(q)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]

    def step(prepared):
        for t in leaves:
            t.grad = None
        attend(*leaves, prepared).backward(gout)

    prep_ms = cuda_ms(prep, iters)
    prepared = prep()
    with torch.no_grad():
        fwd_ms = cuda_ms(lambda: attend(q, k, v, prepared), iters)
    fb_ms = cuda_ms(lambda: step(prepared), iters)

    # one more step from the inputs alone, for its peak memory
    del prepared
    for t in leaves:
        t.grad = None
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    prepared = prep()
    with torch.no_grad():
        attend(q, k, v, prepared)
    step(prepared)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return prep_ms, fwd_ms, fb_ms, prep_ms + fwd_ms + fb_ms, peak


def bench_case(name, parts, hq, hkv, head_dim, baseline, iters):
    blocks, cu = pack(parts)
    n = blocks.shape[0]
    q = torch.randn(n, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    scale = head_dim**-0.5

    new = measure(
        lambda a, b, c, index: qsa_block_attention(a, b, c, index, scale),
        lambda: build_qsa_block_index(blocks, cu, compress_ratio=RATIO, num_heads=hq, num_kv_heads=hkv),
        q,
        k,
        v,
        iters,
    )
    row = (
        f"{hq:>2}/{hkv} {name:>18}  new prep {new[0]:7.1f} fwd {new[1]:8.1f} fwd+bwd {new[2]:8.1f} step {new[3]:8.1f}"
    )
    row += f" peak {new[4]:5.2f}"
    if baseline is None:
        return row

    old = measure(
        lambda a, b, c, form: baseline.qsa_block_sparse_attention(a, b, c, *form, scale, RATIO),
        lambda: bitmap_form(blocks, cu),
        q,
        k,
        v,
        iters,
    )
    row += f" | old prep {old[0]:7.1f} fwd {old[1]:8.1f} fwd+bwd {old[2]:8.1f} step {old[3]:8.1f} peak {old[4]:5.2f}"
    return row + f" | step x{old[3] / new[3]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default="", help="comma-separated files with {'blocks': int32 [T, K]}")
    ap.add_argument("--baseline", default="", help="a kernel file exposing qsa_block_sparse_attention")
    ap.add_argument("--heads", default="12/1,24/2,6/1,3/1", help="TP-local query heads / KV heads")
    ap.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--packs", default="2048+2048+4096", help="'+'-joined lengths packed into one micro-batch")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    if args.selections:
        sources = [torch.load(p)["blocks"].cuda()[:, :BUDGET].contiguous() for p in args.selections.split(",")]
    else:
        longest = max(int(x) for x in args.lengths.split(","))
        sources = [random_blocks(longest, seed=0), random_blocks(longest, seed=1)]
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location("qsa_baseline", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    cases = [(f"{n}", [sources[0][:n]]) for n in map(int, args.lengths.split(",")) if n <= sources[0].shape[0]]
    for spec_ in filter(None, args.packs.split(",")):
        parts = [sources[i % len(sources)][: int(n)] for i, n in enumerate(spec_.split("+"))]
        cases.append((f"pack {spec_}", parts))

    print(f"{torch.cuda.get_device_name()}, head_dim {args.head_dim}, ms; peak GiB", flush=True)
    for heads in args.heads.split(","):
        hq, hkv = map(int, heads.split("/"))
        for name, parts in cases:
            print(bench_case(name, parts, hq, hkv, args.head_dim, baseline, args.iters), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
