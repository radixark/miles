"""Region-sparse attention must match dense attention masked to the selection, forward and backward."""

from tests.ci.ci_register import register_cuda_ci

# tuned for, and only run on, Blackwell: the chunk sizes are what its tensor memory leaves room for
register_cuda_ci(est_time=60, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from miles.kernels.attention.region_sparse import (
    HEAD_DIM,
    REGION_SIZE,
    query_blocks,
    region_unions,
    sparse_region_attention,
    token_region_bitmap,
)
from miles.kernels.attention.region_sparse.layout import ENTRY_REGION_SHIFT, QUERY_BLOCK

# a model's selection budget: the scored regions plus the token's own
TOP_REGIONS = 512
MAX_SELECTED_REGIONS = TOP_REGIONS + 1
HEADS_PER_GROUP = 16


@dataclass(frozen=True)
class RegionSelection:
    region_ids: Tensor  # int32 [T, G, MAX_SELECTED_REGIONS], ascending, -1 padded, own region last
    cu_seqlens: Tensor  # int32 [num_sequences + 1]


SCALE = HEAD_DIM**-0.5


def make_selection(
    lengths: list[int], num_groups: int, budget: int, generator: torch.Generator, recent_bias: float = 0.0
) -> RegionSelection:
    """Per (token, group): ``budget`` history regions picked at random, then the token's own region.

    ``recent_bias`` tilts the picks towards the regions just before the token's own.
    """
    cu = torch.tensor([0, *lengths]).cumsum(0)
    rows = []
    for length in lengths:
        num_regions = -(-length // REGION_SIZE)
        own = torch.arange(length, device="cuda") // REGION_SIZE
        region = torch.arange(num_regions, device="cuda")
        scores = torch.rand(length, num_groups, num_regions, device="cuda", generator=generator)
        distance = (own[:, None] - region[None, :]).clamp_min(0).float()
        scores = scores + recent_bias * torch.exp(-distance / 256.0)[:, None, :]
        scores = torch.where(region[None, None, :] < own[:, None, None], scores, -1.0)
        top = scores.topk(min(budget, num_regions), dim=-1)
        picked = torch.where(top.values >= 0, top.indices, torch.iinfo(torch.int64).max).sort(dim=-1).values
        ids = torch.full((length, num_groups, MAX_SELECTED_REGIONS), -1, dtype=torch.int64, device="cuda")
        ids[:, :, : picked.shape[-1]] = torch.where(picked < num_regions, picked, -1)
        count = torch.clamp(own, max=budget)
        ids.scatter_(2, count[:, None, None].expand(-1, num_groups, 1), own[:, None, None].expand(-1, num_groups, 1))
        rows.append(ids)
    return RegionSelection(region_ids=torch.cat(rows).to(torch.int32), cu_seqlens=cu.to(torch.int32).cuda())


def make_qkv(num_tokens: int, num_groups: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    shape_q = (num_tokens, num_groups * HEADS_PER_GROUP, HEAD_DIM)
    q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(num_tokens, num_groups, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(num_tokens, num_groups, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    return q, k, v


def reference_attention(
    q: Tensor, k: Tensor, v: Tensor, selection: RegionSelection, dout: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """fp32 dense attention masked to the selection, and its gradients for ``dout``.

    One (sequence, group) at a time: attention never crosses either, and a whole-pack score
    tensor would not fit.
    """
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(out)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros_like(dk)
    bounds = selection.cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        length = end - start
        num_regions = -(-length // REGION_SIZE)
        pos = torch.arange(length, device=q.device)
        for g in range(k.shape[1]):
            ids = selection.region_ids[start:end, g].long()
            chosen = torch.zeros(length, num_regions + 1, dtype=torch.bool, device=q.device)
            chosen.scatter_(1, torch.where(ids >= 0, ids, num_regions), True)
            visible = chosen[:, pos // REGION_SIZE] & (pos[None, :] <= pos[:, None])
            heads = slice(g * HEADS_PER_GROUP, (g + 1) * HEADS_PER_GROUP)
            qs = q[start:end, heads].float().requires_grad_()
            ks = k[start:end, g].float().requires_grad_()
            vs = v[start:end, g].float().requires_grad_()
            scores = torch.einsum("thd,sd->hts", qs, ks) * SCALE
            probs = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
            o = torch.einsum("hts,sd->thd", probs, vs)
            grads = torch.autograd.grad(o, (qs, ks, vs), dout[start:end, heads].float())
            out[start:end, heads] = o.detach()
            dq[start:end, heads], dk[start:end, g], dv[start:end, g] = grads
    return out, dq, dk, dv


def sdpa_causal(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
    """bf16 dense causal attention per sequence."""
    outs = []
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        o = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
            is_causal=True,
            scale=SCALE,
            enable_gqa=True,
        )
        outs.append(o.transpose(0, 1))
    return torch.cat(outs)


def run_kernel(q, k, v, selection, dout):
    qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
    out = sparse_region_attention(qq, kk, vv, selection.region_ids, selection.cu_seqlens, SCALE)
    out.backward(dout)
    return out.detach(), qq.grad, kk.grad, vv.grad


def rel_err(got: Tensor, want: Tensor) -> float:
    """Frobenius error relative to the reference's norm."""
    return ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)).item()


def assert_matches_reference(q, k, v, selection, generator):
    dout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    assert_close(run_kernel(q, k, v, selection, dout), reference_attention(q, k, v, selection, dout))


def assert_close(got, want):
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want, strict=True):
        assert a.dtype == torch.bfloat16, name
        err = rel_err(a, b)
        # bf16 rounding of the inputs to each matmul puts a correct kernel near 3e-3
        assert err < 1e-2, (name, err)
        worst = (a.float() - b).abs().max().item()
        assert worst < 5e-2 * b.abs().max().item() + 1e-2, (name, worst)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
@pytest.mark.parametrize("budget", [4, 48])
def test_random_selection_skipping_regions(num_groups, budget):
    g = torch.Generator(device="cuda").manual_seed(budget * 10 + num_groups)
    lengths = [1500]
    selection = make_selection(lengths, num_groups, budget, g)
    q, k, v = make_qkv(sum(lengths), num_groups, g)
    assert_matches_reference(q, k, v, selection, g)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
def test_packed_mixed_lengths(num_groups):
    """Sequences on and off the region grid, and an empty one; the 5000-token one is sparse past position 4103."""
    g = torch.Generator(device="cuda").manual_seed(num_groups)
    lengths = [1, 7, 8, 0, 9, 600, 4104, 5000]
    selection = make_selection(lengths, num_groups, TOP_REGIONS, g, recent_bias=0.5)
    q, k, v = make_qkv(sum(lengths), num_groups, g)
    assert_matches_reference(q, k, v, selection, g)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
def test_every_region_selected_is_causal_attention(num_groups):
    """At most 4104 tokens every region is selected, and the result is dense causal attention."""
    g = torch.Generator(device="cuda").manual_seed(100 + num_groups)
    lengths = [4104, 333, 1024]
    selection = make_selection(lengths, num_groups, TOP_REGIONS, g)
    counts = (selection.region_ids >= 0).sum(-1)
    starts = torch.repeat_interleave(selection.cu_seqlens[:-1].long(), torch.tensor(lengths, device="cuda"))
    pos = torch.arange(sum(lengths), device="cuda") - starts
    assert torch.equal(counts, (pos // REGION_SIZE + 1)[:, None].expand_as(counts).to(counts.dtype))
    q, k, v = make_qkv(sum(lengths), num_groups, g)

    dout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    got = run_kernel(q, k, v, selection, dout)
    qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
    dense = sdpa_causal(qq, kk, vv, selection.cu_seqlens)
    dense.backward(dout)
    want = reference_attention(q, k, v, selection, dout)
    for name, a, b, exact in zip(
        ("out", "dq", "dk", "dv"), got, (dense.detach(), qq.grad, kk.grad, vv.grad), want, strict=True
    ):
        # about as accurate as bf16 causal attention, and in agreement with it
        assert rel_err(a, exact) < 2 * rel_err(b, exact) + 1e-3, (name, rel_err(a, exact), rel_err(b, exact))
        assert rel_err(a, b) < 1e-2, (name, rel_err(a, b))


def test_sequence_past_int32_offsets():
    """With Step5's 64 query heads on one rank, a sequence packed after 174,762 tokens lies past 2^31 q elements."""
    g = torch.Generator(device="cuda").manual_seed(3)
    num_groups, filler, length = 4, 174_768, 40
    assert filler * num_groups * HEADS_PER_GROUP * HEAD_DIM > 2**31
    tail = make_selection([length], num_groups, TOP_REGIONS, g)
    # the filler tokens see only their own region, which keeps its selection small to build
    ids = torch.full((filler + length, num_groups, MAX_SELECTED_REGIONS), -1, dtype=torch.int32, device="cuda")
    ids[:filler, :, 0] = (torch.arange(filler, dtype=torch.int32, device="cuda") // REGION_SIZE)[:, None]
    ids[filler:] = tail.region_ids
    cu = torch.tensor([0, filler, filler + length], dtype=torch.int32, device="cuda")
    q, k, v = make_qkv(filler + length, num_groups, g)
    dout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    got = run_kernel(q, k, v, RegionSelection(region_ids=ids, cu_seqlens=cu), dout)
    s = slice(filler, None)
    assert_close([t[s] for t in got], reference_attention(q[s], k[s], v[s], tail, dout[s]))


def test_layout_lists_each_blocks_union():
    """Blocks tile every sequence from its first token; each lists its tokens' regions once, tagged by token."""
    g = torch.Generator(device="cuda").manual_seed(7)
    lengths = [1, 9, 300, 1000, 0, 17]
    selection = make_selection(lengths, 2, 16, g)
    blocks = query_blocks(selection.cu_seqlens, sum(lengths), MAX_SELECTED_REGIONS)
    unions = region_unions(blocks, token_region_bitmap(selection.region_ids))
    ids = selection.region_ids.cpu()
    first, count, offset = blocks.first_token.cpu(), blocks.num_tokens.cpu(), blocks.union_offset.cpu()
    union_count, entries = unions.count.cpu(), unions.entries.cpu()

    starts = torch.tensor([0, *lengths]).cumsum(0)[:-1]
    want_blocks = [
        (s + p, min(QUERY_BLOCK, n - p))
        for s, n in zip(starts.tolist(), lengths, strict=True)
        for p in range(0, n, QUERY_BLOCK)
    ]
    real = count > 0
    assert list(zip(first[real].tolist(), count[real].tolist(), strict=True)) == want_blocks

    for group in range(ids.shape[1]):
        for block in torch.nonzero(real).flatten().tolist():
            start = group * blocks.group_stride + offset[block]
            got = entries[start : start + union_count[group, block]].tolist()
            masks: dict[int, int] = {}
            for tok in range(int(count[block])):
                for region in ids[first[block] + tok, group].tolist():
                    if region >= 0:
                        masks[region] = masks.get(region, 0) | 1 << tok
            assert got == [region << ENTRY_REGION_SHIFT | mask for region, mask in sorted(masks.items())]


def test_unions_fit_past_4104_regions():
    """Deep in a long sequence a block's union is bounded by 513 regions per token; unions still never overlap."""
    lengths = [33_003, 5, 0, 40_000]
    blocks = query_blocks(
        torch.tensor([0, *lengths]).cumsum(0).to(torch.int32).cuda(), sum(lengths), MAX_SELECTED_REGIONS
    )
    capacity = [
        min(b + 1, min(QUERY_BLOCK, n - b * QUERY_BLOCK) * MAX_SELECTED_REGIONS)
        for n in lengths
        for b in range(-(-n // QUERY_BLOCK))
    ]
    offset = blocks.union_offset[blocks.num_tokens > 0].tolist()
    ends = [start + size for start, size in zip(offset, capacity, strict=True)]
    assert all(end <= start for end, start in zip(ends[:-1], offset[1:], strict=True))
    assert ends[-1] <= blocks.group_stride
