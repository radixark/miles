"""Trainer GDN core: serving's chunk prefill forward, FLA's chunk rule backward.

The value is serving's GDN, which under TOP decodes as its fixed-tile chunk prefill would (rollout decode follows
the chunk-64 prefill bit for bit), so every packed document is one chunk prefill from zero state. The gradient is
that of FLA's chunk gated delta rule over the same documents: the two differ only by rounding. Under context
parallelism a document piece may start from the previous rank's state (a leaf whose gradient goes back to that
rank) and end in the state the next rank continues from.
"""

import functools
import itertools
from typing import NamedTuple

import torch
import torch.nn.functional as F

from miles_plugins.top.kernels.surrogate import reference_mode, surrogate


def _gdn_gating(a, b, A_log, dt_bias):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, b.float().sigmoid()


def _reference_inputs(mixed_qkv, a, b, A_log, dt_bias, heads):
    num_k_heads, head_k_dim, num_v_heads, head_v_dim = heads
    key_dim = num_k_heads * head_k_dim
    t = mixed_qkv.shape[0]
    q, k, v = mixed_qkv.split((key_dim, key_dim, num_v_heads * head_v_dim), dim=-1)
    group = num_v_heads // num_k_heads
    q = q.reshape(1, t, num_k_heads, head_k_dim).repeat_interleave(group, dim=2)
    k = k.reshape(1, t, num_k_heads, head_k_dim).repeat_interleave(group, dim=2)
    v = v.reshape(1, t, num_v_heads, head_v_dim)
    g, beta = _gdn_gating(a, b, A_log, dt_bias)
    return q, k, v, g[None], beta[None].to(v.dtype)


def _reference(q, k, v, g, beta, *, cu_seqlens, lengths=None, initial=None, final_doc=None):
    """FLA chunk rule over the packed documents, each from zero state or (``initial`` = (document, FLA state
    [1, HV, K, V])) one from a given state. q/k/v [1, T, HV, D]; g/beta [1, T, HV]. Returns ([1, T, HV, V], the
    final FLA state of ``final_doc`` or None)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    init = None
    if initial is not None:
        doc, state = initial
        init = q.new_zeros((cu_seqlens.numel() - 1, *state.shape[1:]), dtype=torch.float32)
        init = init.index_copy(0, _ints([doc], q), state)
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=init,
        output_final_state=final_doc is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.long(),
        cu_seqlens_cpu=None if lengths is None else _host_cu(lengths),  # FLA reads lengths here instead of syncing
    )
    return out, (final[final_doc : final_doc + 1] if final_doc is not None else None)


def _forward(
    mixed_qkv, a, b, A_log, dt_bias, *, cu_seqlens, heads, initial_docs=None, initial_states=None, final_docs=None
):
    """Serving's chunk prefill (its TOP kernel object's ``extend``, as serving prefills) over every packed document
    from zero state, or from ``initial_states`` [n, HV, V, K] for ``initial_docs``. Returns [1, T, HV, V]; with
    ``final_docs``, also those documents' final FP32 states [n, HV, V, K]."""
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.true_on_policy.gdn_backend import top_gdn_kernel

    num_k_heads, head_k_dim, num_v_heads, head_v_dim = heads
    t, key_dim = mixed_qkv.shape[0], num_k_heads * head_k_dim
    q, k, v = mixed_qkv.split((key_dim, key_dim, num_v_heads * head_v_dim), dim=-1)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    docs = cu_seqlens.numel() - 1
    state = mixed_qkv.new_zeros((docs, num_v_heads, head_v_dim, head_k_dim), dtype=torch.float32)
    if initial_docs is not None:
        state.index_copy_(0, initial_docs, initial_states.to(torch.float32))
    # Updates state in place, as serving does with its pool slots.
    out = top_gdn_kernel().extend(
        q.view(1, t, num_k_heads, head_k_dim),
        k.view(1, t, num_k_heads, head_k_dim),
        v.view(1, t, num_v_heads, head_v_dim),
        g,
        beta,
        ssm_states=state,
        cache_indices=torch.arange(docs, device=mixed_qkv.device, dtype=torch.int32),
        query_start_loc=cu_seqlens,
    )[0]
    return (out, state.index_select(0, final_docs)) if final_docs is not None else out


class _Group(NamedTuple):
    """Contiguous local documents [first, end) on rows [lo, hi), run as one call."""

    first: int
    end: int
    lo: int
    hi: int
    cu: torch.Tensor  # int32 document bounds within the group
    lengths: tuple


class _ChainPlan(NamedTuple):
    """A rank's documents in row order: the incoming piece, the documents in between, the outgoing piece. Groups that
    do not exist are None; ``incoming`` is ``outgoing`` when one piece both continues and is continued."""

    incoming: _Group | None
    middle: _Group | None
    outgoing: _Group | None


def _chain_plan(layout, lengths, device) -> _ChainPlan:
    """Cached on the layout per document lengths, so serving's FLA index cache sees the same bounds tensors in every
    layer."""
    cache = layout.__dict__.setdefault("_gdn_chain_plans", {})
    if lengths in cache:
        return cache[lengths]
    starts = [0, *itertools.accumulate(lengths)]

    def group(first, end):
        if first >= end:
            return None
        cu = [s - starts[first] for s in starts[first : end + 1]]
        return _Group(first, end, starts[first], starts[end], _ints(cu, device, torch.int32), lengths[first:end])

    n = len(lengths)
    inc = 0 if layout.incoming is not None else None
    out = n - 1 if layout.outgoing is not None else None
    if inc is not None and inc == out:
        single = group(0, n)
        plan = _ChainPlan(single, None, single)
    else:
        lo = 1 if inc is not None else 0
        hi = n - 1 if out is not None else n
        plan = _ChainPlan(
            group(0, 1) if inc is not None else None,
            group(lo, hi),
            group(n - 1, n) if out is not None else None,
        )
    cache[lengths] = plan
    return plan


class _ContextParallelCore(torch.autograd.Function):
    """Serving's forward with the previous rank's state in and this rank's outgoing state out; reference backward
    with the incoming state as a leaf, its gradient sent back, and the outgoing state's gradient received.

    Each rank runs its documents as up to three calls so that the chain only waits on what it needs: forward runs the
    outgoing piece first and sends its state at once, then the documents in between, then the incoming piece once
    the previous rank's state has arrived; backward runs the incoming piece first and sends its state's gradient,
    then the documents in between, then the outgoing piece once its gradient has arrived. Serving's chunk prefill
    and FLA's chunk rule compute every document on its own, so the split changes no byte; without it, every rank
    waited for the whole GDN core of the ranks before it (forward) or after it (backward)."""

    @staticmethod
    def forward(ctx, mixed_qkv, a, b, A_log, dt_bias, cfg):
        from miles_plugins.top.gdn_cp import post_p2p, wait_p2p

        layout, heads = cfg["layout"], cfg["heads"]
        shape = (1, heads[2], heads[3], heads[1])  # serving state [doc, HV, V, K]
        plan = _chain_plan(layout, cfg["lengths"], mixed_qkv.device)
        hit, init = cfg["cached"]
        recv = None if hit or layout.prev is None else mixed_qkv.new_empty(shape, dtype=torch.float32)
        zero = _ints([0], mixed_qkv)

        def run(group, state=None, final=False):
            rows = slice(group.lo, group.hi)
            states = {}
            if state is not None:
                states.update(initial_docs=zero, initial_states=state)
            if final:
                states.update(final_docs=_ints([group.end - group.first - 1], mixed_qkv))
            return cfg["forward"](mixed_qkv[rows], a[rows], b[rows], A_log, dt_bias, cu_seqlens=group.cu, **states)

        outs = {}
        with torch.no_grad():
            reqs = []
            if plan.outgoing is not None and plan.outgoing is not plan.incoming:
                outs["outgoing"], final = run(plan.outgoing, final=True)
                if not hit:
                    reqs = post_p2p(send=final, to=layout.next, recv=recv, frm=layout.prev, group=layout.group)
            elif recv is not None:
                reqs = post_p2p(recv=recv, frm=layout.prev, group=layout.group)
            if plan.middle is not None:
                outs["middle"] = run(plan.middle)
            wait_p2p(reqs)
            if recv is not None:
                init = recv
            if cfg["keep"]:
                layout.stash.keep((cfg["key"], "state"), init)
            if plan.incoming is not None:
                if plan.incoming is plan.outgoing:
                    outs["incoming"], final = run(plan.incoming, init, final=True)
                    if not hit:
                        wait_p2p(post_p2p(send=final, to=layout.next, group=layout.group))
                else:
                    outs["incoming"] = run(plan.incoming, init)
        parts = [outs[k] for k in ("incoming", "middle", "outgoing") if k in outs]
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        ctx.cfg, ctx.init, ctx.plan = cfg, init, plan
        ctx.save_for_backward(mixed_qkv, a, b, A_log, dt_bias)
        return out

    @staticmethod
    def backward(ctx, d_out):
        from miles_plugins.top.gdn_cp import post_p2p, wait_p2p

        cfg, layout, plan = ctx.cfg, ctx.cfg["layout"], ctx.plan
        heads = cfg["heads"]
        inputs = ctx.saved_tensors
        leaves = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
        d_final = None
        if layout.next is not None:
            d_final = inputs[0].new_empty((1, heads[2], heads[3], heads[1]), dtype=torch.float32)
        init = ctx.init.detach().requires_grad_() if ctx.init is not None else None
        with torch.enable_grad(), reference_mode():
            ref_inputs = _reference_inputs(*leaves, heads)
        # FLA's chunk rule per group on detached slices; the gating and head expansion run backward once, over the
        # concatenated slice gradients (a concatenation, as autograd's split backward: no adds).
        parts = [x.detach() for x in ref_inputs]

        def grad(group, state=None, d_state=None):
            rows = slice(group.lo, group.hi)
            xs = [x[:, rows].requires_grad_() for x in parts]
            final_doc = group.end - group.first - 1 if d_state is not None else None
            with torch.enable_grad(), reference_mode():
                out, final = _reference(
                    *xs,
                    cu_seqlens=group.cu,
                    lengths=group.lengths,
                    initial=(0, state.transpose(-1, -2)) if state is not None else None,
                    final_doc=final_doc,
                )
                outputs, grads = [out], [d_out[:, rows]]
                if final is not None:
                    outputs.append(final.transpose(-1, -2))
                    grads.append(d_state)
                wrt = xs + ([state] if state is not None else [])
                return torch.autograd.grad(outputs, wrt, grads)

        got, reqs = {}, []
        if plan.incoming is not None:
            if plan.incoming is plan.outgoing:
                wait_p2p(post_p2p(recv=d_final, frm=layout.next, group=layout.group) if d_final is not None else [])
                got["incoming"] = grad(plan.incoming, init, d_final)
            else:
                got["incoming"] = grad(plan.incoming, init)
            send = got["incoming"][-1] if init is not None else None
            recv = d_final if plan.outgoing is not None and plan.outgoing is not plan.incoming else None
            reqs = post_p2p(send=send, to=layout.prev, recv=recv, frm=layout.next, group=layout.group)
        elif d_final is not None:
            reqs = post_p2p(recv=d_final, frm=layout.next, group=layout.group)
        if plan.middle is not None:
            got["middle"] = grad(plan.middle)
        wait_p2p(reqs)
        if plan.outgoing is not None and plan.outgoing is not plan.incoming:
            got["outgoing"] = grad(plan.outgoing, d_state=d_final)
        order = [got[k][:5] for k in ("incoming", "middle", "outgoing") if k in got]
        d_ref = [g[0] if len(order) == 1 else torch.cat(g, dim=1) for g in zip(*order, strict=True)]
        pairs = [(r, d) for r, d in zip(ref_inputs, d_ref, strict=True) if r.requires_grad]
        if pairs:
            with torch.enable_grad(), reference_mode():
                torch.autograd.backward(*zip(*pairs, strict=True))
        return (*(t.grad if t.requires_grad else None for t in leaves), None)


def _host_cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.long)


def _ints(values, like, dtype=torch.int64):
    from miles_plugins.top.host_copy import device_tensor

    return device_tensor(values, device=like if isinstance(like, torch.device) else like.device, dtype=dtype)


def delegated_gdn(mixed_qkv, a, b, A_log, dt_bias, *, cu_seqlens, heads, lengths=None, cp=None, key=None):
    """GDN gating and core: serving's chunk prefill over each packed document as the value, FLA's chunk rule as the
    gradient. ``mixed_qkv`` [T, 2*H*K + HV*V] post-conv, raw gates ``a``/``b`` [T, HV]; ``heads`` = (H, K, HV, V);
    ``lengths`` the documents' host lengths. Under context parallelism ``cp`` (gdn_cp) chains the state across ranks;
    ``key`` (the calling layer) lets a recompute reuse the first pass's incoming state. Returns [1, T, HV, V]."""
    forward = functools.partial(_forward, cu_seqlens=cu_seqlens, heads=heads)
    if cp is not None:
        if lengths is None:
            bounds = cu_seqlens.tolist()
            lengths = tuple(b - a for a, b in itertools.pairwise(bounds))
        # A recompute reuses the first pass's incoming state and skips the exchange (gdn_cp.RecomputeStash).
        cached = cp.stash.take((key, "state")) if key is not None else (False, None)
        keep = key is not None and cp.stash.keeping()
        cfg = dict(
            forward=forward,
            heads=heads,
            cu_seqlens=cu_seqlens,
            lengths=lengths,
            layout=cp,
            cached=cached,
            keep=keep,
            key=key,
        )
        return _ContextParallelCore.apply(mixed_qkv, a, b, A_log, dt_bias, cfg)

    def reference(mixed_qkv, a, b, A_log, dt_bias):
        q, k, v, g, beta = _reference_inputs(mixed_qkv, a, b, A_log, dt_bias, heads)
        return _reference(q, k, v, g, beta, cu_seqlens=cu_seqlens, lengths=lengths)[0]

    return surrogate(forward, reference, mixed_qkv, a, b, A_log, dt_bias, name="gdn_core")
