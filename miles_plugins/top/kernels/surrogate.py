"""Surrogate gradients for delegated forwards that keep no autograd state.

The forward value is the serving kernel's (the bytes rollout produced). The gradient is
that of a PyTorch reference of the same math, recomputed in backward at the saved inputs.
Backward needs ordinary accuracy only, so the reference only has to agree to rounding.
"""

from __future__ import annotations

import contextlib
import os

import torch

# TOP_SURROGATE_CHECK=1 keeps the forward value and compares the reference against it in backward.
_CHECK = os.environ.get("TOP_SURROGATE_CHECK") == "1"
_CHECK_TOL = float(os.environ.get("TOP_SURROGATE_CHECK_TOL", "0.05"))


@contextlib.contextmanager
def reference_mode():
    """References are ordinary PyTorch: the serving batch-invariant overrides stay off."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode

    with set_batch_invariant_mode(False):
        yield


def _versions(inputs):
    # Saved-tensor hooks (activation offload) snapshot the inputs at save time, so later in-place changes cannot
    # reach backward, and the unpacked copies carry fresh version counters: nothing to check.
    if torch._C._autograd._top_saved_tensors_default_hooks(False) is not None:
        return None
    return [t._version for t in inputs]


def _check_versions(ctx, inputs):
    if ctx.versions is not None and [t._version for t in inputs] != ctx.versions:
        raise RuntimeError(f"[top] {ctx.name}: an input changed between forward and backward")


class _Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_fn, reference_fn, name, *inputs):
        ctx.reference_fn, ctx.name = reference_fn, name
        ctx.versions = _versions(inputs)
        ctx.save_for_backward(*inputs)
        with torch.no_grad():
            out = visible_fn(*(t.detach() for t in inputs))
        single = isinstance(out, torch.Tensor)
        outs = (out,) if single else tuple(out)
        ctx.single = single
        ctx.visible = tuple(o.detach() for o in outs) if _CHECK else None
        for o in outs:
            if not o.is_floating_point():
                ctx.mark_non_differentiable(o)
        return out if single else outs

    @staticmethod
    def backward(ctx, *grads):
        inputs = ctx.saved_tensors
        _check_versions(ctx, inputs)
        leaves = [t.detach().requires_grad_(t.requires_grad and t.is_floating_point()) for t in inputs]
        with torch.enable_grad(), reference_mode():
            ref = ctx.reference_fn(*leaves)
        refs = (ref,) if isinstance(ref, torch.Tensor) else tuple(ref)
        if ctx.visible is not None:
            for i, (r, v) in enumerate(zip(refs, ctx.visible, strict=True)):
                if v.is_floating_point() and v.numel():
                    err = float((r.detach().float() - v.float()).norm() / v.float().norm().clamp_min(1e-30))
                    if not err < _CHECK_TOL:
                        raise RuntimeError(
                            f"[top] {ctx.name} output {i}: reference differs from the kernel by {err:.3e}"
                        )
        pairs = [(r, g) for r, g in zip(refs, grads, strict=True) if g is not None and r.requires_grad]
        want = [i for i, t in enumerate(leaves) if t.requires_grad]
        result = [None] * len(leaves)
        if pairs and want:
            with reference_mode():
                got = torch.autograd.grad(
                    [r for r, _ in pairs], [leaves[i] for i in want], [g for _, g in pairs], allow_unused=True
                )
            for i, g in zip(want, got, strict=True):
                result[i] = g
        return (None, None, None, *result)


def surrogate(visible_fn, reference_fn, *inputs, name="surrogate"):
    """visible_fn(*inputs) under no_grad gives the value; reference_fn(*inputs) gives the gradient.

    All inputs are tensors (pass constants by closure); parameters must be passed as inputs to
    receive gradients."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in inputs):
        with torch.no_grad():
            return visible_fn(*(t.detach() for t in inputs))
    return _Surrogate.apply(visible_fn, reference_fn, name, *inputs)


class _LinearSurrogate(torch.autograd.Function):
    """``visible_fn(x, weight)``'s value with the gradient of ``x @ value.T``, in closed form; ``value`` is
    ``value_fn()`` (the served weight, rebuilt in backward instead of saved) or ``weight`` itself. The weight gradient
    ``grad.T @ x`` does not depend on the value and goes to ``weight`` (straight through when they differ).

    surrogate() would rebuild the reference forward GEMM in backward only to differentiate it."""

    @staticmethod
    def forward(ctx, visible_fn, name, value_fn, x, weight):
        ctx.name, ctx.value_fn = name, value_fn
        ctx.versions = _versions((x, weight))
        ctx.save_for_backward(x, weight)
        with torch.no_grad():
            out = visible_fn(x.detach(), weight.detach())
        ctx.visible = out.detach() if _CHECK else None
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        _check_versions(ctx, (x, weight))
        x2, g2 = x.reshape(-1, x.shape[-1]), grad.reshape(-1, grad.shape[-1])
        needs_value = ctx.needs_input_grad[3] or ctx.visible is not None
        value = ctx.value_fn() if ctx.value_fn is not None and needs_value else weight
        with reference_mode():
            if ctx.visible is not None and ctx.visible.numel():
                v = ctx.visible.reshape(-1, ctx.visible.shape[-1]).float()
                err = float(((x2 @ value.t()).float() - v).norm() / v.norm().clamp_min(1e-30))
                if not err < _CHECK_TOL:
                    raise RuntimeError(f"[top] {ctx.name}: reference differs from the kernel by {err:.3e}")
            grad_x = (g2 @ value).view(x.shape) if ctx.needs_input_grad[3] else None
            grad_w = g2.t() @ x2 if ctx.needs_input_grad[4] else None
        return None, None, None, grad_x, grad_w


def linear_surrogate(visible_fn, x, weight, name="linear", value_fn=None):
    """surrogate() for y = x @ value.T: same gradients, without recomputing the forward GEMM. ``value_fn()`` gives the
    weight the backward is evaluated at (the served weight of a quantized master, rebuilt in backward); without it,
    ``weight``."""
    if not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad):
        with torch.no_grad():
            return visible_fn(x.detach(), weight.detach())
    return _LinearSurrogate.apply(visible_fn, name, value_fn, x, weight)


class _RowSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible_fn, chunk_reference_fn, chunk_rows, name, chunks, chunk_backward_fn, x, *params):
        ctx.chunk_reference_fn, ctx.chunk_rows, ctx.name, ctx.chunks = chunk_reference_fn, chunk_rows, name, chunks
        # The closed form skips the reference; the kernel-vs-reference check needs the reference.
        ctx.chunk_backward_fn = None if _CHECK else chunk_backward_fn
        ctx.versions = _versions((x, *params))
        ctx.save_for_backward(x, *params)
        with torch.no_grad():
            out = visible_fn(x.detach(), *(p.detach() for p in params))
        ctx.visible = out.detach() if _CHECK else None
        return out

    @staticmethod
    def backward(ctx, grad):
        result = _RowSurrogate._backward(ctx, grad)
        # The reference closures can hold decoded weights (the served experts) that the graph would keep until freed.
        ctx.chunk_reference_fn = ctx.chunk_backward_fn = None
        return result

    @staticmethod
    def _backward(ctx, grad):
        x, *params = ctx.saved_tensors
        _check_versions(ctx, (x, *params))
        leaves = [p.detach().requires_grad_(p.requires_grad) for p in params]
        if x.shape[0] <= ctx.chunk_rows:
            return (None, None, None, None, None, None, *_row_backward_one_chunk(ctx, x, leaves, grad))
        if ctx.chunks is not None:
            return (None, None, None, None, None, None, *_row_backward_planned(ctx, x, params, leaves, grad))
        grad_x = torch.zeros_like(x) if x.requires_grad else None
        # Weight gradients sum over chunks; accumulate in FP32 so chunking adds no BF16 rounding.
        grad_params = [torch.zeros_like(p, dtype=torch.float32) if p.requires_grad else None for p in params]
        err_sq = ref_sq = 0.0
        for lo in range(0, x.shape[0], ctx.chunk_rows):
            hi = min(lo + ctx.chunk_rows, x.shape[0])
            x_chunk = x[lo:hi].detach().requires_grad_(x.requires_grad)
            with torch.enable_grad(), reference_mode():
                ref = ctx.chunk_reference_fn(lo, hi, x_chunk, *leaves)
            if ctx.visible is not None:
                v = ctx.visible[lo:hi].float()
                err_sq += float((ref.detach().float() - v).square().sum())
                ref_sq += float(v.square().sum())
            wrt = ([x_chunk] if x.requires_grad else []) + [p for p in leaves if p.requires_grad]
            with reference_mode():
                got = list(torch.autograd.grad(ref, wrt, grad[lo:hi], allow_unused=True))
            if x.requires_grad:
                grad_x[lo:hi] = got.pop(0)
            for acc, g in zip([a for a in grad_params if a is not None], got, strict=True):
                if g is not None:
                    acc += g
        if ctx.visible is not None and ref_sq > 0 and not (err_sq / ref_sq) ** 0.5 < _CHECK_TOL:
            raise RuntimeError(
                f"[top] {ctx.name}: reference differs from the kernel by {(err_sq / ref_sq) ** 0.5:.3e}"
            )
        result = [None if a is None else a.to(p.dtype) for a, p in zip(grad_params, params, strict=True)]
        return (None, None, None, None, None, None, grad_x, *result)


def _row_backward_one_chunk(ctx, x, leaves, grad):
    """All rows in one chunk: autograd's gradients directly, no FP32 accumulation buffers."""
    if ctx.chunk_backward_fn is not None:
        with torch.no_grad(), reference_mode():
            grad_x, grads = ctx.chunk_backward_fn(0, x.shape[0], x.detach(), grad, *leaves)
        return (
            grad_x if x.requires_grad else None,
            *[g if p.requires_grad else None for g, p in zip(grads, leaves, strict=True)],
        )
    x_leaf = x.detach().requires_grad_(x.requires_grad)
    with torch.enable_grad(), reference_mode():
        ref = ctx.chunk_reference_fn(0, x.shape[0], x_leaf, *leaves)
    if ctx.visible is not None and ctx.visible.numel():
        v = ctx.visible.float()
        err = float((ref.detach().float() - v).norm() / v.norm().clamp_min(1e-30))
        if not err < _CHECK_TOL:
            raise RuntimeError(f"[top] {ctx.name}: reference differs from the kernel by {err:.3e}")
    wrt = ([x_leaf] if x.requires_grad else []) + [p for p in leaves if p.requires_grad]
    with reference_mode():
        got = list(torch.autograd.grad(ref, wrt, grad, allow_unused=True))
    grad_x = got.pop(0) if x.requires_grad else None
    grad_params = [got.pop(0) if p.requires_grad else None for p in leaves]
    return (grad_x, *grad_params)


def _row_backward_planned(ctx, x, params, leaves, grad):
    """Chunks from a plan of (lo, hi, indices of the params those rows touch). A param touched by one
    chunk takes that chunk's gradient as is; only params split across chunks accumulate in FP32."""
    touches = [0] * len(params)
    for _, _, touched in ctx.chunks:
        for i in touched:
            touches[i] += 1
    grad_x = torch.empty_like(x) if x.requires_grad else None
    result = [None] * len(params)
    err_sq = ref_sq = 0.0
    for lo, hi, touched in ctx.chunks:
        want = [i for i in sorted(touched) if params[i].requires_grad]
        if ctx.chunk_backward_fn is not None:
            with torch.no_grad(), reference_mode():
                chunk_grad_x, chunk_grads = ctx.chunk_backward_fn(lo, hi, x[lo:hi].detach(), grad[lo:hi], *leaves)
            if x.requires_grad:
                grad_x[lo:hi] = chunk_grad_x
            got = [chunk_grads[i] for i in want]
        else:
            x_chunk = x[lo:hi].detach().requires_grad_(x.requires_grad)
            with torch.enable_grad(), reference_mode():
                ref = ctx.chunk_reference_fn(lo, hi, x_chunk, *leaves)
            if ctx.visible is not None:
                v = ctx.visible[lo:hi].float()
                err_sq += float((ref.detach().float() - v).square().sum())
                ref_sq += float(v.square().sum())
            wrt = ([x_chunk] if x.requires_grad else []) + [leaves[i] for i in want]
            with reference_mode():
                got = list(torch.autograd.grad(ref, wrt, grad[lo:hi], allow_unused=True))
            if x.requires_grad:
                grad_x[lo:hi] = got.pop(0)
        for i, g in zip(want, got, strict=True):
            if g is None:
                continue
            if touches[i] == 1:
                # g views this chunk's gradient of every stacked param; a copy frees the rest.
                result[i] = g.clone()
            elif result[i] is None:
                result[i] = g.float()
            else:
                result[i] += g
    if ctx.visible is not None and ref_sq > 0 and not (err_sq / ref_sq) ** 0.5 < _CHECK_TOL:
        raise RuntimeError(f"[top] {ctx.name}: reference differs from the kernel by {(err_sq / ref_sq) ** 0.5:.3e}")
    # Untouched params (an expert with no rows) get zeros, as the unplanned path gives them.
    return (
        grad_x,
        *[
            (torch.zeros_like(p) if r is None else r.to(p.dtype)) if p.requires_grad else None
            for r, p in zip(result, params, strict=True)
        ],
    )


def row_surrogate(
    visible_fn, chunk_reference_fn, x, *params, chunk_rows, name="row_surrogate", chunks=None, chunk_backward_fn=None
):
    """surrogate() for a reference that is independent per row of ``x``: backward recomputes it
    ``chunk_rows`` rows at a time, so its peak memory does not grow with a hot expert's row count.

    chunk_reference_fn(lo, hi, x[lo:hi], *params) gives those rows' reference output. ``chunks`` is an
    optional plan of (lo, hi, param indices those rows touch) covering all rows in order; with it, a
    param touched by one chunk gets its gradient once instead of an FP32 sum over every chunk.
    ``chunk_backward_fn(lo, hi, x[lo:hi], grad[lo:hi], *params)`` -> (grad_x, [grad per param]) is an optional
    closed form of the reference's backward, used instead of recomputing the reference and differentiating it."""
    if not torch.is_grad_enabled() or not (x.requires_grad or any(p.requires_grad for p in params)):
        with torch.no_grad():
            return visible_fn(x.detach(), *(p.detach() for p in params))
    return _RowSurrogate.apply(visible_fn, chunk_reference_fn, chunk_rows, name, chunks, chunk_backward_fn, x, *params)


def checkpointed(reference_fn, *inputs):
    """``reference_fn(*inputs)``, keeping only its inputs: backward recomputes it, the same ops on the same inputs, so
    the same gradient, instead of holding its intermediates until backward (under pipeline parallelism, across every
    in-flight micro-batch). For the references attach() takes; reference_fn must have no side effects."""
    from torch.utils.checkpoint import checkpoint

    return checkpoint(reference_fn, *inputs, use_reentrant=False, preserve_rng_state=False)


def attach(visible, reference, name="attach"):
    """visible's value with reference's gradient, for references cheap enough to keep in forward.

    reference - reference.detach() is exactly zero, so the value is visible's bytes."""
    if not torch.is_grad_enabled() or not reference.requires_grad:
        return visible
    reference = reference.to(visible.dtype).view(visible.shape)
    return visible.detach() + (reference - reference.detach())


def rms_norm_reference(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def fp8_group_fake_quant(x, group, scale="ue8m0"):
    """E4M3 with one scale per ``group`` elements, as straight-through: the value is rounded, the gradient is identity.
    ``scale`` is serving's rule: "ue8m0", a power of two (DeepSeek-V4's KV), or "amax", the FP32 amax / 448 (the DSA
    FP8 KV cache, quant_k_cache)."""
    shape = x.shape
    g = x.float().reshape(-1, group)
    amax = g.abs().amax(-1, keepdim=True)
    if scale == "ue8m0":
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448.0)))
    elif scale == "amax":
        # An all-zero group stays zero (serving divides 0 by 0 there and never reads it back as nonzero).
        s = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    else:
        raise ValueError(f"[top] unknown FP8 scale rule {scale}")
    q = (g / s).to(torch.float8_e4m3fn).float() * s
    return (g + (q - g).detach()).reshape(shape).to(x.dtype)
