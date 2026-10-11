import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK_SIZE = 32
COMPRESSED_KV_BLOCK_SIZE = 16
FP8_AMAX_FLOOR = 1e-4
FP4_AMAX_FLOOR = 6 * 2.0**-126
COMPRESSED_KV_SCALE_MIN = 2.0**-9


@triton.jit
def _ceil_pow2(v):
    bits = v.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(tl.int32)
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fake_quant_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    FMAX: tl.constexpr,
    INV_FMAX: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    SCALE_MAX: tl.constexpr,
    FP4: tl.constexpr,
    E4M3_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows < n_rows)[:, None]
    offs = rows[:, None] * BLOCK + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if E4M3_SCALE:
        scale = tl.minimum(tl.maximum(amax * INV_FMAX, AMAX_FLOOR), SCALE_MAX)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        scaled = tl.div_rn(x, scale[:, None])
    else:
        amax = tl.maximum(amax, AMAX_FLOOR)
        scale = _ceil_pow2(amax * INV_FMAX)
        scaled = x / scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -FMAX), FMAX)
    if FP4:
        mag = tl.abs(scaled)
        step = tl.where(mag < 2.0, 0.5, tl.where(mag < 4.0, 1.0, 2.0))
        sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
        q = libdevice.rint(mag / step) * step * sign
    else:
        q = scaled.to(tl.float8e4nv).to(tl.float32)
    out = q * scale[:, None]
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _fake_quant(x: torch.Tensor, fp4: bool, compressed_kv: bool = False) -> torch.Tensor:
    block = COMPRESSED_KV_BLOCK_SIZE if compressed_kv else BLOCK_SIZE
    assert x.shape[-1] % block == 0
    xc = x.contiguous()
    out = torch.empty_like(xc)
    n_rows = xc.numel() // block
    rows_per_prog = 64
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    if compressed_kv:
        fmax, floor = FP4_MAX, COMPRESSED_KV_SCALE_MIN
    else:
        fmax, floor = (FP4_MAX, FP4_AMAX_FLOOR) if fp4 else (FP8_MAX, FP8_AMAX_FLOOR)
    _fake_quant_kernel[grid](
        xc,
        out,
        n_rows,
        FMAX=fmax,
        INV_FMAX=1.0 / fmax,
        AMAX_FLOOR=floor,
        SCALE_MAX=FP8_MAX,
        FP4=fp4,
        E4M3_SCALE=compressed_kv,
        BLOCK=block,
        ROWS=rows_per_prog,
    )
    return out


def fake_quant_compressed_kv_forward(x: torch.Tensor) -> torch.Tensor:
    return _fake_quant(x, True, compressed_kv=True)


def fake_quant_fp4_forward(x: torch.Tensor) -> torch.Tensor:
    return _fake_quant(x, True)


def fake_quant_fp8_forward(x: torch.Tensor) -> torch.Tensor:
    return _fake_quant(x, False)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn):
        return fn(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    return _StraightThrough.apply(x, fake_quant_fp8_forward)


def fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    return _StraightThrough.apply(x, fake_quant_fp4_forward)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    return _StraightThrough.apply(x, fake_quant_compressed_kv_forward)
