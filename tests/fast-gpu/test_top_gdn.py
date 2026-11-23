"""The GDN TOP trainer core is serving's chunk prefill, bit for bit, with FLA's chunk rule as its backward.

Qwen3.5-35B-A3B GDN shapes. The packed forward equals serving prefilling each request alone (its qkv split, gating
and TOP kernel object); a document resumed from its state on the 64-row grid (an exact-CP piece) continues it
exactly; the gradient is the FLA reference's. Under CP, each rank's state chain runs its documents in up to three
calls (the piece a neighbour waits for first) and changes no byte: every rank's output is CP1's, and every gradient,
including the incoming state's handed back, is one FLA reference call over the rank's documents. Those CP ranks run in
one process, forward in rank order and backward in reverse, with the point-to-point exchange replaced by a mailbox.
"""

import itertools
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.top import gdn_cp  # noqa: E402
from miles_plugins.top.kernels.gdn import _forward, _reference, _reference_inputs, delegated_gdn  # noqa: E402

HEADS = (16, 128, 32, 128)  # H, K, HV, V
LENGTHS = (1, 3, 63, 64, 65, 200, 333)


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _inputs(rows, seed=0):
    h, k, hv, v = HEADS
    g = torch.Generator(device="cuda").manual_seed(seed)
    mixed = torch.randn(rows, 2 * h * k + hv * v, generator=g, device="cuda").to(torch.bfloat16)
    a = (torch.randn(rows, hv, generator=g, device="cuda") * 2).to(torch.bfloat16)
    b = (torch.randn(rows, hv, generator=g, device="cuda") * 2).to(torch.bfloat16)
    A_log = (torch.rand(hv, generator=g, device="cuda") * 2 - 1).float()
    dt_bias = (torch.randn(hv, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    return mixed, a, b, A_log, dt_bias


def _serving_prefill(mixed, a, b, A_log, dt_bias):
    """One request prefilled from an empty slot as serving's GDN backend runs it (forward_extend): fused qkv split,
    fused gating, its TOP kernel object's extend into the pool slot. Returns the output [L, HV, V] and the slot."""
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
    from sglang.srt.true_on_policy.gdn_backend import top_gdn_kernel

    h, k, hv, v = HEADS
    q, key, value = fused_qkv_split_gdn_prefill(mixed, h, h, hv, k, k, v)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    slot = torch.zeros(1, hv, v, k, device="cuda", dtype=torch.float32)
    out = top_gdn_kernel().extend(
        q,
        key,
        value,
        g,
        beta,
        ssm_states=slot,
        cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        query_start_loc=_cu([mixed.shape[0]]),
    )[0]
    return out[0], slot[0]


def _same(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)


def test_packed_forward_is_serving_prefill_per_request():
    inputs = _inputs(sum(LENGTHS))
    with torch.no_grad():
        out = delegated_gdn(*inputs, cu_seqlens=_cu(LENGTHS), heads=HEADS, lengths=LENGTHS)
    starts = [0, *itertools.accumulate(LENGTHS)]
    for lo, hi in itertools.pairwise(starts):
        expected, _ = _serving_prefill(*(x[lo:hi] for x in inputs[:3]), *inputs[3:])
        assert _same(out[0, lo:hi], expected), f"rows [{lo}, {hi})"


@pytest.mark.parametrize("cut", [64, 128, 256])
def test_a_piece_resumed_on_the_split_grid_continues_the_document(cut):
    """Exact CP: the previous rank's piece hands its final state on; the next piece starts from it."""
    length = 333
    mixed, a, b, A_log, dt_bias = _inputs(length, seed=2)
    whole, whole_state = _serving_prefill(mixed, a, b, A_log, dt_bias)
    with torch.no_grad():
        first, state = _forward(
            mixed[:cut],
            a[:cut],
            b[:cut],
            A_log,
            dt_bias,
            cu_seqlens=_cu([cut]),
            heads=HEADS,
            final_docs=torch.zeros(1, device="cuda", dtype=torch.long),
        )
        second, final = _forward(
            mixed[cut:],
            a[cut:],
            b[cut:],
            A_log,
            dt_bias,
            cu_seqlens=_cu([length - cut]),
            heads=HEADS,
            initial_docs=torch.zeros(1, device="cuda", dtype=torch.long),
            initial_states=state,
            final_docs=torch.zeros(1, device="cuda", dtype=torch.long),
        )
    assert _same(torch.cat((first[0], second[0])), whole)
    assert _same(final[0], whole_state)


def test_gradient_is_the_fla_reference_gradient():
    inputs = _inputs(sum(LENGTHS), seed=3)
    d_out = torch.randn(1, sum(LENGTHS), HEADS[2], HEADS[3], device="cuda").to(torch.bfloat16)
    grads = []
    for run in ("delegated", "reference"):
        leaves = [x.clone().requires_grad_() for x in inputs]
        if run == "delegated":
            out = delegated_gdn(*leaves, cu_seqlens=_cu(LENGTHS), heads=HEADS, lengths=LENGTHS)
        else:
            q, k, v, g, beta = _reference_inputs(*leaves, HEADS)
            out = _reference(q, k, v, g, beta, cu_seqlens=_cu(LENGTHS), lengths=LENGTHS)[0]
        out.backward(d_out)
        grads.append([x.grad for x in leaves])
    for name, delegated, reference in zip(("mixed_qkv", "a", "b", "A_log", "dt_bias"), *grads, strict=True):
        assert _same(delegated, reference), name


def test_gdn_block_runs_forward_and_backward():
    """TopQwenGDN (conv, core, gated norm, projections) over packed documents: finite output and every gradient."""
    from miles_plugins.top.gdn import TopQwenGDN

    h, k, hv, v = HEADS
    config = SimpleNamespace(
        linear_num_key_heads=h,
        linear_num_value_heads=hv,
        linear_key_head_dim=k,
        linear_value_head_dim=v,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        hidden_size=512,
    )
    torch.manual_seed(4)
    block = TopQwenGDN(config).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for p in block.parameters():
            if p.dtype == torch.bfloat16:
                p.normal_(0, 0.05)
    leaf = torch.randn(sum(LENGTHS), config.hidden_size, device="cuda").to(torch.bfloat16).requires_grad_()
    y = block(leaf, _cu(LENGTHS), lengths_cpu=list(LENGTHS))
    y.backward(torch.randn_like(y))
    assert torch.isfinite(y).all() and torch.isfinite(leaf.grad).all()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def _bits(x):
    x = x.detach().contiguous()
    return x.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()])


def _same_bits(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and torch.equal(_bits(x), _bits(y))


class _Mailbox:
    """post_p2p / wait_p2p for ranks that run one after another in this process."""

    def __init__(self):
        self.box = defaultdict(deque)
        self.me = None

    def post(self, send=None, to=None, recv=None, frm=None, group=None):
        if send is not None:
            self.box[(self.me, to)].append(send.detach().clone())
        if recv is None:
            return []
        me = self.me
        return [lambda: recv.copy_(self.box[(frm, me)].popleft())]

    @staticmethod
    def wait(reqs):
        for req in reqs:
            req()


def _layout(pieces, rank, lengths):
    mine = pieces[rank]
    incoming = 0 if mine[0][1] > 0 else None
    outgoing = len(mine) - 1 if mine[-1][2] < lengths[mine[-1][0]] else None
    return gdn_cp.GDNCPLayout(
        group=None,
        rank=rank,
        size=len(pieces),
        prev=rank - 1 if incoming is not None else None,
        next=rank + 1 if outgoing is not None else None,
        incoming=incoming,
        outgoing=outgoing,
        max_rows=0,
        local_rows=sum(b - a for _, a, b in mine),
        tail_end=0,
        keys=None,
        conv_init=None,
        output_rows=None,
        stash=gdn_cp.RecomputeStash(True),
    )


def _chain_inputs(lengths, seed):
    h, k, hv, v = HEADS
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = sum(lengths)
    mixed = torch.randn(rows, 2 * h * k + hv * v, generator=g, device="cuda").to(torch.bfloat16)
    a = (torch.randn(rows, hv, generator=g, device="cuda") * 2).to(torch.bfloat16)
    b = (torch.randn(rows, hv, generator=g, device="cuda") * 2).to(torch.bfloat16)
    A_log = (torch.rand(hv, generator=g, device="cuda") * 2 - 1).float()
    dt_bias = (torch.randn(hv, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    d_out = torch.randn(1, rows, hv, v, generator=g, device="cuda").to(torch.bfloat16)
    return (mixed, a, b, A_log, dt_bias), d_out


def _local(x, pieces, starts):
    return torch.cat([x[starts[d] + a : starts[d] + b] for d, a, b in pieces])


def _local_rows(pieces, starts):
    return [r for d, a, b in pieces for r in range(starts[d] + a, starts[d] + b)]


CASES = [
    ((1, 3, 63, 64, 65, 200, 333, 700), 2),
    ((1, 3, 63, 64, 65, 200, 333, 700), 3),
    ((1, 3, 63, 64, 65, 200, 333, 700), 4),
    ((2000,), 4),  # middle ranks hold one piece that both continues and is continued
    ((300, 1500, 7), 4),
]


@pytest.fixture
def trainer_fla(monkeypatch):
    """FLA as the TOP trainer runs it: top_launch sets FLA_DISABLE_BACKEND_DISPATCH=1, so FLA's own Triton kernels run.
    FLA reads that flag at import (the test session imports FLA earlier); an empty backend registry has the same
    effect. With dispatch on, a document's gradient bits can depend on the documents beside it in the call."""
    from fla.ops.backends import BackendRegistry

    monkeypatch.setattr(BackendRegistry, "_registries", {})
    monkeypatch.setattr(BackendRegistry, "ensure_initialized", classmethod(lambda cls, operation: None))


@pytest.mark.parametrize("lengths,size", CASES)
def test_chain_forward_is_cp1_and_backward_is_the_one_call_reference(lengths, size, monkeypatch, trainer_fla):
    mail = _Mailbox()
    monkeypatch.setattr(gdn_cp, "post_p2p", mail.post)
    monkeypatch.setattr(gdn_cp, "wait_p2p", mail.wait)
    pieces = gdn_cp.partition(list(lengths), size, grid=64)
    starts = [0, *itertools.accumulate(lengths)]
    inputs, d_out = _chain_inputs(lengths, seed=len(lengths) + size)
    with torch.no_grad():
        cp1 = delegated_gdn(*inputs, cu_seqlens=_cu(lengths), heads=HEADS, lengths=lengths)

    layouts = [_layout(pieces, r, lengths) for r in range(size)]
    local_lengths = [tuple(b - a for _, a, b in p) for p in pieces]
    leaves, outs, kept = [], [], []
    for r in range(size):
        mail.me = r
        x = [_local(t, pieces[r], starts) for t in inputs[:3]] + [t for t in inputs[3:]]
        x = [t.clone().requires_grad_() for t in x]
        cu = _cu(local_lengths[r])
        # Full recompute: the training forward keeps the incoming state; the recompute reuses it.
        with torch.no_grad():
            first = delegated_gdn(*x, cu_seqlens=cu, heads=HEADS, lengths=local_lengths[r], cp=layouts[r], key="L")
        kept.append(layouts[r].stash._kept.get(("L", "state")))
        out = delegated_gdn(*x, cu_seqlens=cu, heads=HEADS, lengths=local_lengths[r], cp=layouts[r], key="L")
        assert _same_bits(out, first), f"rank {r}: recompute differs from the training forward"
        rows = _local_rows(pieces[r], starts)
        assert _same_bits(out[0], cp1[0, rows]), f"rank {r}: forward differs from CP1"
        leaves.append(x)
        outs.append(out)
    for r in reversed(range(size)):
        mail.me = r
        outs[r].backward(d_out[:, _local_rows(pieces[r], starts)])
    assert not any(mail.box.values()), "a message was sent and never received"

    # The chain before the split: one FLA call per rank over all its documents, state in and out.
    d_init_next = None
    for r in reversed(range(size)):
        lay, x = layouts[r], [t.detach().clone().requires_grad_() for t in leaves[r]]
        init = kept[r].detach().clone().requires_grad_() if kept[r] is not None else None
        q, k, v, g, beta = _reference_inputs(*x, HEADS)
        out, final = _reference(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=_cu(local_lengths[r]),
            lengths=local_lengths[r],
            initial=(lay.incoming, init.transpose(-1, -2)) if init is not None else None,
            final_doc=lay.outgoing if lay.next is not None else None,
        )
        outputs, grads = [out], [d_out[:, _local_rows(pieces[r], starts)]]
        if final is not None:
            outputs.append(final.transpose(-1, -2))
            grads.append(d_init_next)
        torch.autograd.backward(outputs, grads)
        for name, got, want in zip(("mixed_qkv", "a", "b", "A_log", "dt_bias"), leaves[r], x, strict=True):
            assert _same_bits(got.grad, want.grad), f"rank {r}: d{name}"
        d_init_next = init.grad if init is not None else None
