"""Off-grid BF16 masters for a quantized release checkpoint, keyed on the weight-sync quantizer's grid.

A release that ships quantized weights gives masters that are its exact dequantization. When weight sync requantizes
with a rule whose grid holds those values, every master sits on a grid point, and round-to-nearest maps any update
smaller than half a cell back to the same code: the served weights stay put for hundreds of steps at lr 1e-6.
``offset_off_grid`` moves each on-grid master once, at load, to a hashed point of its own rounding interval (up to
half a step toward either neighbouring code), so updates cross rounding boundaries in proportion to their size. Every
element whose byte would change is pulled back, so the sync quantizer still returns exactly the unmoved master's codes
and scales.

A grid is the sync quantizer's element format, block shape and scale rule (``SyncGrid``), not the checkpoint's: a
checkpoint is on a grid whenever its values are representable there (MXFP4 experts dequantized into 128x128 block
FP8, say). With power-of-two scales, a block whose maximum is the top code sits on the scale-rounding boundary
(amax / top = 2^e), so any increase of that element doubles the block's scale; ``lower_top`` moves those maxima down
inside their cell. FP32-amax scales follow the maximum continuously, so there the maximum stays where it is.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class SyncGrid(NamedTuple):
    # "e4m3" (FP8) or "e2m1" (FP4).
    element: str
    # Elements per scale: (rows, columns).
    block: tuple[int, int]
    # "ue8m0": power-of-two scales rounded up from amax / top; "fp32_amax": the FP32 scale amax / top.
    scale: str


E4M3_128_UE8M0 = SyncGrid("e4m3", (128, 128), "ue8m0")
E4M3_32_UE8M0 = SyncGrid("e4m3", (32, 32), "ue8m0")
E4M3_128_FP32 = SyncGrid("e4m3", (128, 128), "fp32_amax")
E4M3_32_FP32 = SyncGrid("e4m3", (32, 32), "fp32_amax")
MXFP4 = SyncGrid("e2m1", (1, 32), "ue8m0")
# Declared, not built: NVFP4's per-tensor FP32 global scale moves with the tensor amax, which re-rounds every block
# scale at once, so the masters also need the global amax frozen between steps.
NVFP4 = SyncGrid("e2m1", (1, 16), "e4m3_global")

GRIDS = (E4M3_128_UE8M0, E4M3_32_UE8M0, E4M3_128_FP32, E4M3_32_FP32, MXFP4)
# Grids pinned byte for byte against the real sync quantizer: on GPU for the 128x128 weight-sync grid; by construction
# for DeepSeek-V4.1's 32x32 projections and MXFP4 experts, whose weight sync sends ``requantize``'s own bytes
# (fp8_b32.quantize_b32, dsv41_experts.sync_form). The other GRIDS are exact against this module's model of their
# quantizer only, so they are refused until pinned.
PINNED_GRIDS = (E4M3_128_UE8M0, E4M3_32_UE8M0, MXFP4)

_E4M3_TOP = 448.0
_E2M1_TOP = 6.0
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Distance from each E2M1 level to the level below it and to the level above it.
_E2M1_DOWN = (0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0)
_E2M1_UP = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0)
_ROWS_PER_CHUNK = 64 * 128


def check_grid(grid: SyncGrid) -> None:
    if grid == NVFP4:
        raise NotImplementedError(
            "[top] no off-grid offset for NVFP4: its global scale follows the tensor amax, so the masters also need "
            "the global amax frozen between steps (not built)"
        )
    if grid not in GRIDS:
        raise ValueError(f"[top] no off-grid offset for sync grid {grid}; supported: {GRIDS}")
    if grid not in PINNED_GRIDS:
        raise NotImplementedError(
            f"[top] the off-grid offset for {grid} is not pinned against its GPU sync quantizer; pinned: {PINNED_GRIDS}"
        )


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """SGLang's fp8_utils.ceil_to_ue8m0 (from DeepGEMM): the power of two at or above |x|."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


class _Cells(NamedTuple):
    codes: torch.Tensor  # uint8 per element: the sync quantizer's byte (E2M1: one nibble)
    scale: torch.Tensor  # FP32 per block
    scale_el: torch.Tensor  # FP32 per element
    amax_el: torch.Tensor  # FP32 per element: the block maximum, as the quantizer bounds it
    dequant: torch.Tensor  # FP32 per element


def _blocks(x: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    m, n = x.shape
    return x.view(m // block[0], block[0], n // block[1], block[1])


def _expand(per_block: torch.Tensor, block: tuple[int, int], shape) -> torch.Tensor:
    return per_block.expand(-1, block[0], -1, block[1]).reshape(shape)


def _encode(wf: torch.Tensor, grid: SyncGrid, scale_el: torch.Tensor) -> torch.Tensor:
    """The sync quantizer's byte of each element at fixed block scales."""
    if grid.element == "e4m3":
        if grid.scale == "ue8m0":
            # sglang per_block_cast_to_fp8 (exact: the scale is a power of two).
            return (wf * (1.0 / scale_el)).to(torch.float8_e4m3fn).view(torch.uint8)
        # miles fp8_blockwise_cast: multiply by the FP32 reciprocal, clamp, round to nearest even.
        return (wf * (1.0 / scale_el)).clamp(-_E4M3_TOP, _E4M3_TOP).to(torch.float8_e4m3fn).view(torch.uint8)
    # miles.utils.mxfp4.quantize_mxfp4: round to nearest with ties down, sign kept on zero.
    normalized = wf * (1.0 / scale_el)
    magnitude = torch.zeros_like(normalized, dtype=torch.uint8)
    for bound in _E2M1_BOUNDS:
        magnitude.add_(normalized.abs() > bound)
    return magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)


def _cells(w: torch.Tensor, grid: SyncGrid) -> _Cells:
    """The sync quantizer on a 2-D BF16 chunk whose shape is a multiple of the block."""
    wf = w.float()
    amax = _blocks(wf, grid.block).abs().amax(dim=(1, 3), keepdim=True)
    if grid.element == "e4m3":
        if grid.scale == "ue8m0":
            amax = amax.clamp(1e-4)
            scale = _ceil_to_ue8m0(amax / _E4M3_TOP)
        else:
            amax = amax.clamp(1e-10)
            scale = amax / _E4M3_TOP
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax / _E2M1_TOP)).clamp(-127, 127))
    scale_el = _expand(scale, grid.block, w.shape)
    codes = _encode(wf, grid, scale_el)
    if grid.element == "e4m3":
        value = codes.view(torch.float8_e4m3fn).float()
    else:
        levels = torch.tensor(_E2M1_LEVELS, device=w.device)[(codes & 7).long()]
        value = torch.where((codes & 8) != 0, -levels, levels)
    return _Cells(
        codes,
        scale.view(scale.shape[0], scale.shape[2]),
        scale_el,
        _expand(amax, grid.block, w.shape),
        value * scale_el,
    )


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    h = h & 0xFFFFFFFF
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _block_uniform(w: torch.Tensor) -> torch.Tensor:
    """[0, 1) per element of a 2-D BF16 tensor, hashed from its bits and its position inside a 128x128 tile: equal
    on every DP replica and for any 128-aligned shard or expert stack."""
    m, n = w.shape
    bits = w.view(torch.int16).to(torch.int64) & 0xFFFF
    r = (torch.arange(m, device=w.device, dtype=torch.int64) % 128).view(-1, 1)
    c = (torch.arange(n, device=w.device, dtype=torch.int64) % 128).view(1, -1)
    h = _fmix32(_fmix32(bits | (r << 16) | (c << 23)) ^ 0x2545F491)
    return (h >> 8).to(torch.float32) * (1.0 / (1 << 24))


def _steps(cells: _Cells, grid: SyncGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Each element's distance to the code below it and to the code above it, on the master's scale. They differ at a
    binade boundary (E4M3 powers of two; E2M1 2 and 4), where rounding reaches half a coarser step upward."""
    if grid.element == "e4m3":
        code = cells.codes.view(torch.float8_e4m3fn).float().abs()

        def step(e):
            return torch.ldexp(torch.ones_like(cells.scale_el), (e - 1).clamp_min(-6) - 3) * cells.scale_el

        return step(torch.frexp(code * (1 - 2**-10))[1]), step(torch.frexp(code)[1])
    level = (cells.codes & 7).long()
    down, up = (torch.tensor(t, device=cells.codes.device)[level] for t in (_E2M1_DOWN, _E2M1_UP))
    return down * cells.scale_el, up * cells.scale_el


def _top_code(cells: _Cells, grid: SyncGrid) -> torch.Tensor:
    if grid.element == "e4m3":
        return (cells.codes & 0x7F) == 0x7E  # 448
    return (cells.codes & 7) == 7  # 6


def on_grid(master: torch.Tensor, grid: SyncGrid) -> torch.Tensor:
    """Which elements of a 2-D BF16 master the sync quantizer maps back onto themselves (after rounding to BF16)."""
    check_grid(grid)
    padded, (m, n) = _pad(master, grid)
    cells = _cells(padded, grid)
    return (cells.dequant.to(torch.bfloat16) == padded)[:m, :n]


def requantize(master: torch.Tensor, grid: SyncGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """The sync quantizer's (codes uint8 per element, FP32 scales per block) for a 2-D BF16 master."""
    check_grid(grid)
    padded, (m, n) = _pad(master, grid)
    cells = _cells(padded, grid)
    br, bc = grid.block
    return cells.codes[:m, :n], cells.scale[: -(-m // br), : -(-n // bc)]


def _pad(w: torch.Tensor, grid: SyncGrid) -> tuple[torch.Tensor, tuple[int, int]]:
    """Zero-pad to whole blocks, as the FP8 quantizers do; MXFP4 groups never straddle a row end."""
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise ValueError(f"[top] off-grid masters are 2-D BF16, got {tuple(w.shape)} {w.dtype}")
    m, n = w.shape
    br, bc = grid.block
    if grid.element == "e2m1" and n % bc:
        raise ValueError(f"[top] {n} columns are not a whole number of {bc}-element MXFP4 groups")
    pm, pn = -(-m // br) * br, -(-n // bc) * bc
    if (pm, pn) == (m, n):
        return w, (m, n)
    padded = w.new_zeros((pm, pn))
    padded[:m, :n] = w
    return padded, (m, n)


def _offset_chunk(w: torch.Tensor, grid: SyncGrid, lower_top: bool) -> torch.Tensor:
    wf = w.float()
    cells = _cells(w, grid)
    down, up = _steps(cells, grid)
    h = _block_uniform(w)
    # A hashed point of the element's rounding interval, in magnitude: below it by up to half the step down, above
    # it by up to half the step up. Zeros (~12% of a release's MXFP4 experts) move toward their own sign, so their
    # byte holds.
    magnitude = (h - 0.5) * torch.where(h >= 0.5, up, down)
    magnitude = torch.where(cells.dequant == 0, magnitude.abs(), magnitude)
    moved = (wf + torch.where(torch.signbit(wf), -magnitude, magnitude)).to(torch.bfloat16)
    moved = torch.where(moved.float().abs() > cells.amax_el, w, moved)
    at_max = wf.abs() == cells.amax_el
    if grid.scale == "fp32_amax":
        moved = torch.where(at_max, w, moved)
    elif lower_top:
        top = at_max & _top_code(cells, grid)
        moved = torch.where(top, (wf - torch.sign(wf) * 0.5 * h * down).to(torch.bfloat16), moved)
    moved = torch.where(cells.dequant.to(torch.bfloat16) == w, moved, w)
    for _ in range(8):
        changed = _encode(moved.float(), grid, cells.scale_el) != cells.codes
        if not changed.any():
            break
        moved = torch.where(changed, (wf + 0.5 * (moved.float() - wf)).to(torch.bfloat16), moved)
    moved = torch.where(changed, w, moved)
    # Byte-pinned per element at the old scales; a block whose own scale would move goes back whole.
    after = _cells(moved, grid)
    scale_moved = (after.scale != cells.scale).view(after.scale.shape[0], 1, after.scale.shape[1], 1)
    if scale_moved.any():
        moved = torch.where(_expand(scale_moved, grid.block, w.shape), w, moved)
        after = _cells(moved, grid)
    if not (torch.equal(after.codes, cells.codes) and torch.equal(after.scale, cells.scale)):
        raise AssertionError(f"[top] off-grid offset changed the sync quantizer's bytes on {grid}")
    return moved


def offset_off_grid(master: torch.Tensor, grid: SyncGrid, *, lower_top: bool = True) -> torch.Tensor:
    """``master`` ([..., rows, columns] BF16) with every element that sits on ``grid`` moved inside its rounding
    interval; the sync quantizer returns the same codes and scales for the result. Off-grid elements are returned
    unchanged."""
    check_grid(grid)
    if master.dtype != torch.bfloat16:
        raise ValueError(f"[top] off-grid masters are BF16, got {master.dtype}")
    out = torch.empty_like(master)
    flat_in = master.reshape(-1, *master.shape[-2:])
    flat_out = out.view(-1, *master.shape[-2:])
    for matrix, result in zip(flat_in, flat_out, strict=True):
        for r0 in range(0, matrix.shape[0], _ROWS_PER_CHUNK):
            chunk = matrix[r0 : r0 + _ROWS_PER_CHUNK]
            padded, (m, n) = _pad(chunk, grid)
            result[r0 : r0 + m] = _offset_chunk(padded, grid, lower_top)[:m, :n]
    return out
