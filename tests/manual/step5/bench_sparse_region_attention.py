"""Benchmark sparse region attention against dense causal attention.

Beyond the dense threshold, use captured selections or a random selection biased
toward recent regions. The baseline is Transformer Engine DotProductAttention.
"""

import argparse
from collections.abc import Callable

import torch
import triton
from torch import Tensor
from transformer_engine.pytorch import DotProductAttention

from miles.kernels.attention.region_sparse import (
    HEAD_DIM,
    query_blocks,
    region_unions,
    sparse_region_attention,
    token_region_bitmap,
)
from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS, RegionSelection

HEADS_PER_GROUP = 16
SCALE = HEAD_DIM**-0.5


def recent_biased_selection(num_tokens: int, num_groups: int, generator: torch.Generator) -> Tensor:
    """512 history regions per (token, group), drawn with weight decaying away from the token, plus its own."""
    own = torch.arange(num_tokens, device="cuda") // REGION_SIZE
    num_regions = int(own[-1]) + 1
    ids = torch.full((num_tokens, num_groups, MAX_SELECTED_REGIONS), -1, dtype=torch.int32, device="cuda")
    region = torch.arange(num_regions, device="cuda")
    for start in range(0, num_tokens, 1024):
        rows = own[start : start + 1024]
        distance = (rows[:, None] - region[None, :]).float()
        weight = torch.where(distance > 0, 1.0 / (1.0 + distance / 64.0), 0.0)
        noise = torch.rand(rows.shape[0], num_groups, num_regions, device="cuda", generator=generator)
        # Gumbel-top-k: sampling without replacement in proportion to the weight
        scores = torch.where(weight[:, None, :] > 0, weight.log()[:, None, :] - (-noise.log()).log(), -torch.inf)
        top = scores.topk(min(TOP_REGIONS, num_regions), dim=-1)
        picked = torch.where(top.values > -torch.inf, top.indices, num_regions).sort(dim=-1).values
        picked = torch.where(picked < num_regions, picked, -1)
        count = torch.clamp(rows, max=TOP_REGIONS)
        block = ids[start : start + rows.shape[0]]
        block[:, :, : picked.shape[-1]] = picked
        block.scatter_(
            2,
            count[:, None, None].expand(-1, num_groups, 1).long(),
            rows[:, None, None].expand(-1, num_groups, 1).int(),
        )
    return ids


def make_selection(num_tokens: int, num_groups: int, captured: Tensor | None, generator) -> RegionSelection:
    cu = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")
    if captured is not None:
        assert captured.shape[0] >= num_tokens and captured.shape[1] >= num_groups
        # the capture is causal, so its prefix is the selection of the shorter sequence
        ids = captured[:num_tokens, :num_groups].cuda().contiguous()
    else:
        ids = recent_biased_selection(num_tokens, num_groups, generator)
    return RegionSelection(region_ids=ids, cu_seqlens=cu)


def timed(fn: Callable[[], object]) -> float:
    return triton.testing.do_bench(fn, warmup=5, rep=40)


def bench_backward(out: Tensor, inputs: tuple[Tensor, ...]) -> float:
    grad = torch.randn_like(out)
    return timed(lambda: torch.autograd.grad(out, inputs, grad, retain_graph=True))


def peak_memory(fn: Callable[[], Tensor], inputs: tuple[Tensor, ...]) -> tuple[float, float]:
    """(MiB kept for backward besides the output, MiB peak above the inputs over forward plus backward)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    saved = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
    torch.autograd.grad(out, inputs, torch.ones_like(out))
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return saved / 2**20, peak / 2**20


def run_case(num_tokens: int, num_groups: int, captured: Tensor | None) -> dict[str, float]:
    g = torch.Generator(device="cuda").manual_seed(0)
    heads = num_groups * HEADS_PER_GROUP
    q = torch.randn(num_tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_()
    k = torch.randn(num_tokens, num_groups, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(num_tokens, num_groups, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k.requires_grad_()
    v.requires_grad_()
    inputs = (q, k, v)
    selection = make_selection(num_tokens, num_groups, captured, g)

    row = {}
    blocks = query_blocks(selection.cu_seqlens, num_tokens, MAX_SELECTED_REGIONS)
    unions = region_unions(blocks, token_region_bitmap(selection.region_ids))
    real = blocks.num_tokens > 0
    # the share of the walked (token, region) pairs that the token selected
    walked = (unions.count[:, real].sum(0) * blocks.num_tokens[real]).sum()
    row["useful"] = ((selection.region_ids >= 0).sum() / walked).item()
    sparse = lambda: sparse_region_attention(q, k, v, selection.region_ids, selection.cu_seqlens, SCALE)  # noqa: E731
    row["sparse fwd ms"] = timed(sparse)
    row["sparse bwd ms"] = bench_backward(sparse(), inputs)
    row["sparse saved MiB"], row["sparse peak MiB"] = peak_memory(sparse, inputs)

    cu = selection.cu_seqlens
    te_attn = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=HEAD_DIM,
        num_gqa_groups=num_groups,
        attn_mask_type="padding_causal",
        qkv_format="thd",
        softmax_scale=SCALE,
    )
    dense = lambda: te_attn(  # noqa: E731
        q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=num_tokens, max_seqlen_kv=num_tokens
    )
    row["dense fwd ms"] = timed(dense)
    row["dense bwd ms"] = bench_backward(dense(), inputs)
    row["dense saved MiB"], row["dense peak MiB"] = peak_memory(dense, inputs)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--selection", default=None, help="torch file with a captured region_ids [T, G, 513]")
    args = parser.parse_args()
    captured = torch.load(args.selection)["region_ids"] if args.selection else None

    header = None
    for num_tokens in args.tokens:
        for num_groups in args.groups:
            row = run_case(num_tokens, num_groups, captured)
            if header is None:
                header = ["T", "G", *row]
                print(" | ".join(header))
            print(
                " | ".join([str(num_tokens), str(num_groups), *(f"{value:.4g}" for value in row.values())]), flush=True
            )


if __name__ == "__main__":
    main()
