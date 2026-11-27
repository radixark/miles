"""Clamped SwiGLU forward and backward in one pass each, bit for bit the eager FP32 formulas of fp8_forward.

Every operation rounds as its eager torch counterpart: FMA contraction and libdevice's flush-to-zero are off, silu is x / (1 + expf(-x)) and the sigmoid
1 / (1 + expf(-x)) with correctly rounded divides, clamps propagate NaN like torch.clamp, and products keep the eager association. Only the FP32 temporaries
go away.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024


@triton.jit
def _swiglu_forward_kernel(gate_up, out, half, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < half
    gate = tl.load(gate_up + row * 2 * half + cols, mask=live, other=0.0).to(tl.float32)
    up = tl.load(gate_up + row * 2 * half + half + cols, mask=live, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit, propagate_nan=tl.PropagateNan.ALL)
        up = tl.minimum(
            tl.maximum(up, -limit, propagate_nan=tl.PropagateNan.ALL), limit, propagate_nan=tl.PropagateNan.ALL
        )
    silu = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(out + row * half + cols, (silu * up).to(out.dtype.element_ty), mask=live)


def swiglu_forward(gate_up, swiglu_limit):
    """silu(clamped gate) * clamped up for contiguous [rows, 2I] BF16 gate/up, as [rows, I] BF16."""
    rows, width = gate_up.shape
    out = gate_up.new_empty((rows, width // 2))
    if rows:
        _swiglu_forward_kernel[(rows, triton.cdiv(width // 2, _BLOCK))](
            gate_up,
            out,
            width // 2,
            float(swiglu_limit if swiglu_limit is not None else 0.0),
            HAS_LIMIT=swiglu_limit is not None,
            BLOCK=_BLOCK,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out


@triton.jit
def _swiglu_backward_kernel(gate_up, grad, out, half, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < half
    base = row * 2 * half
    gate = tl.load(gate_up + base + cols, mask=live, other=0.0).to(tl.float32)
    up = tl.load(gate_up + base + half + cols, mask=live, other=0.0).to(tl.float32)
    g = tl.load(grad + row * half + cols, mask=live, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        # Open interval: torch.clamp's backward zeroes the gradient at the bound itself.
        gate_pass = gate < limit
        up_pass = (up > -limit) & (up < limit)
        gate = tl.minimum(gate, limit, propagate_nan=tl.PropagateNan.ALL)
        up = tl.minimum(
            tl.maximum(up, -limit, propagate_nan=tl.PropagateNan.ALL), limit, propagate_nan=tl.PropagateNan.ALL
        )
    sig = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    d_gate = (g * up) * (sig * (1.0 + gate * (1.0 - sig)))
    d_up = g * (gate * sig)
    if HAS_LIMIT:
        # Eager multiplies by the mask (x * 0 keeps NaN and the sign of zero).
        d_gate = tl.where(gate_pass, d_gate, d_gate * 0.0)
        d_up = tl.where(up_pass, d_up, d_up * 0.0)
    tl.store(out + base + cols, d_gate.to(out.dtype.element_ty), mask=live)
    tl.store(out + base + half + cols, d_up.to(out.dtype.element_ty), mask=live)


def swiglu_backward(gate_up, grad, swiglu_limit):
    """d(gate_up) for contiguous [rows, 2I] BF16 gate/up and [rows, I] output grad."""
    rows, width = gate_up.shape
    out = torch.empty_like(gate_up)
    if rows:
        _swiglu_backward_kernel[(rows, triton.cdiv(width // 2, _BLOCK))](
            gate_up,
            grad.contiguous(),
            out,
            width // 2,
            float(swiglu_limit if swiglu_limit is not None else 0.0),
            HAS_LIMIT=swiglu_limit is not None,
            BLOCK=_BLOCK,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out
