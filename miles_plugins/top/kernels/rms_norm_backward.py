"""Analytic RMSNorm adjoint with fixed-order, tiled weight-gradient reduction.

Reconstruct the FP32 residual sum from its operands, not the rounded output.
The weight partials are 1/32 of an activation-sized FP32 temporary; no atomics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _input_grad(X, R, W, DY, DR, DX, INV, H: tl.constexpr, EPS: tl.constexpr,
                HAS_R: tl.constexpr, HAS_DY: tl.constexpr, HAS_DR: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    z = tl.load(X + row * H + cols, mask, 0).to(tl.float32)
    if HAS_R:
        z += tl.load(R + row * H + cols, mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
    tl.store(INV + row, inv)
    grad = tl.full((BLOCK,), 0, tl.float32)
    if HAS_DY:
        dy = tl.load(DY + row * H + cols, mask, 0).to(tl.float32)
        w = tl.load(W + cols, mask, 0).to(tl.float32)
        u = dy * w
        grad = inv * (u - z * (inv * inv / H) * tl.sum(u * z, 0))
    if HAS_DR:
        grad += tl.load(DR + row * H + cols, mask, 0).to(tl.float32)
    tl.store(DX + row * H + cols, grad, mask)


@triton.jit
def _weight_partials(X, R, DY, INV, PARTIAL, M: tl.constexpr, H: tl.constexpr,
                     HAS_R: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    offsets = rows[:, None] * H + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    z = tl.load(X + offsets, mask, 0).to(tl.float32)
    if HAS_R:
        z += tl.load(R + offsets, mask, 0).to(tl.float32)
    dy = tl.load(DY + offsets, mask, 0).to(tl.float32)
    inv = tl.load(INV + rows, rows < M, 0)
    dw = tl.sum(dy * z * inv[:, None], axis=0)
    tl.store(PARTIAL + tl.program_id(0) * H + cols, dw, cols < H)


@triton.jit
def _weight_reduce(PARTIAL, DW, H: tl.constexpr, N: tl.constexpr,
                   BLOCK_N: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, BLOCK_N)
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    acc = tl.full((BLOCK_N, COLS), 0, tl.float32)
    for start in range(tl.cdiv(N, BLOCK_N)):
        r = rows + start * BLOCK_N
        acc += tl.load(PARTIAL + r[:, None] * H + cols[None, :],
                       (r[:, None] < N) & (cols[None, :] < H), 0)
    tl.store(DW + cols, tl.sum(acc, axis=0), cols < H)


def rms_norm_backward(x, weight, eps, grad_output, *, residual=None, grad_residual=None):
    """Return dx, dr, dw using the usual floating-point analytic adjoint."""
    x = x.contiguous()
    residual = residual.contiguous() if residual is not None else None
    dy = grad_output.contiguous() if grad_output is not None else None
    dr = grad_residual.contiguous() if grad_residual is not None else None
    h = x.shape[-1]
    rows = x.numel() // h
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight)
    if rows == 0:
        return dx, dx if residual is not None else None, dw
    inv = torch.empty(rows, device=x.device, dtype=torch.float32)
    _input_grad[(rows,)](
        x, residual, weight, dy, dr, dx, inv, h, eps,
        residual is not None, dy is not None, dr is not None,
        triton.next_power_of_2(h), enable_fp_fusion=False,
    )
    if dy is not None:
        groups = triton.cdiv(rows, 32)
        partial = torch.empty((groups, h), device=x.device, dtype=torch.float32)
        _weight_partials[(groups, triton.cdiv(h, 128))](
            x, residual, dy, inv, partial, rows, h, residual is not None,
            32, 128, enable_fp_fusion=False,
        )
        _weight_reduce[(triton.cdiv(h, 32),)](
            partial, dw, h, groups, min(triton.next_power_of_2(groups), 256), 32,
        )
    return dx, dx if residual is not None else None, dw
