import math

import pytest
import torch

from miles.utils.mxfp4 import quantize_mxfp4
from miles_plugins.top import off_grid
from miles_plugins.top.fp8_masters import SYNC_GRID
from miles_plugins.top.off_grid import (
    E4M3_32_UE8M0,
    E4M3_128_UE8M0,
    GRIDS,
    MXFP4,
    NVFP4,
    PINNED_GRIDS,
    SyncGrid,
    offset_off_grid,
    on_grid,
    requantize,
)


@pytest.fixture(autouse=True)
def every_modelled_grid(monkeypatch):
    """These tests check the module's model of every grid; production admits only the pinned ones."""
    monkeypatch.setattr(off_grid, "PINNED_GRIDS", GRIDS)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequant(master, grid):
    codes, scales = requantize(master, grid)
    br, bc = grid.block
    scale = scales.repeat_interleave(br, 0)[: codes.shape[0]].repeat_interleave(bc, 1)[:, : codes.shape[1]]
    if grid.element == "e4m3":
        return codes.view(torch.float8_e4m3fn).float() * scale
    value = _E2M1[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -value, value) * scale


def _release_master(shape, grid, seed=0):
    """A master as a quantized release gives it: the exact dequantization of codes on ``grid`` (|w| median ~0.016,
    as DeepSeek-V4.1's dense and expert weights)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(shape, generator=g) * 0.023).to(torch.bfloat16)
    return _dequant(w, grid).to(torch.bfloat16)


@pytest.mark.parametrize("grid", GRIDS, ids=str)
@pytest.mark.parametrize("lower_top", [False, True])
def test_requantized_bytes_are_unchanged(grid, lower_top):
    """Step 0: the sync quantizer returns the unmoved master's codes and scales bit for bit (zero signs included)."""
    master = _release_master((300, 512), grid)
    assert bool(on_grid(master, grid).all())
    moved = offset_off_grid(master, grid, lower_top=lower_top)
    for before, after in zip(requantize(master, grid), requantize(moved, grid), strict=True):
        assert torch.equal(before, after)
    assert (moved != master).float().mean() > 0.85


def test_off_grid_elements_stay():
    """Only masters the sync quantizer maps back onto themselves move."""
    g = torch.Generator().manual_seed(1)
    master = (torch.randn(256, 256, generator=g) * 0.023).to(torch.bfloat16)
    moved = offset_off_grid(master, E4M3_128_UE8M0)
    assert torch.equal(moved[~on_grid(master, E4M3_128_UE8M0)], master[~on_grid(master, E4M3_128_UE8M0)])


def test_requantize_is_the_sync_quantizers():
    """The grid model is the weight-sync quantizer's arithmetic: SGLang's UE8M0 block cast (padding included), and
    for MXFP4 Miles' packer away from rounding ties. TOP's MXFP4 sync (dsv41_experts) is this model, which rounds a
    tie toward zero; Miles' packer rounds it to even and is not on TOP's path. (FP32-amax mirrors
    fp8_blockwise_cast, a Triton kernel with no CPU path.)"""
    fp8_utils = pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    g = torch.Generator().manual_seed(2)
    w = (torch.randn(300, 400, generator=g) * 0.023).to(torch.bfloat16)
    codes, scales = requantize(w, E4M3_128_UE8M0)
    ref_codes, ref_scales = fp8_utils.per_block_cast_to_fp8(w)
    assert torch.equal(codes, ref_codes.view(torch.uint8)) and torch.equal(scales, ref_scales)

    w = w[:, :384]
    codes, scales = requantize(w, MXFP4)
    packed, exponents = quantize_mxfp4(w, 32)
    assert torch.equal(torch.log2(scales).to(torch.int32) + 127, exponents.to(torch.int32))
    magnitude = (w.float() / scales.repeat_interleave(32, dim=1)).abs()
    tie = torch.isin(magnitude, torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
    packer_codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(codes.shape)
    assert torch.equal(codes[~tie], packer_codes[~tie])


def _carry(master, grid, update):
    """Share of ``update`` that reaches the served weights: <dq(bf16(m + U)) - dq(m), U> / |U|^2."""
    base = _dequant(master, grid)
    step = _dequant((master.float() + update).to(torch.bfloat16), grid)
    return float(((step - base) * update).sum() / (update * update).sum())


@pytest.mark.parametrize(
    ("release", "grid"),
    [(E4M3_128_UE8M0, E4M3_128_UE8M0), (E4M3_32_UE8M0, E4M3_32_UE8M0), (MXFP4, MXFP4), (MXFP4, E4M3_128_UE8M0)],
    ids=["e4m3-128", "e4m3-32", "mxfp4", "mxfp4-served-as-e4m3-128"],
)
def test_offset_carries_small_updates(release, grid):
    """Apply 100 fixed-sign 1e-6 steps as one update: on-grid masters carry ~0 of it to serving (release: 0.081
    dense FP8 32x32, 0.000 MXFP4; the E4M3 rows' zeros carry from the start); offset masters most of it (BF16 ceiling
    0.95). MXFP4 experts served as 128x128 FP8 sit mostly on E4M3 powers of two, whose rounding interval reaches half
    a coarser step upward: an offset drawn from the finer step on both sides carries 0.48 there, this one 0.66."""
    master = _release_master((512, 1024), release)
    assert bool(on_grid(master, grid).all())
    sign = torch.randn(master.shape, generator=torch.Generator().manual_seed(3)).sign()
    moved = offset_off_grid(master, grid)
    stalled, carried = _carry(master, grid, 100e-6 * sign), _carry(moved, grid, 100e-6 * sign)
    assert stalled < 0.15
    assert 0.6 < carried < 1.0


def _warmup_cosine_total_lr(steps=50):
    """Summed lr of a warmup-cosine schedule: lr 1e-5, linear warmup over 10 steps, cosine to 1e-6 over 300."""
    lr = [1e-5 * t / 10 if t <= 10 else 1e-6 + 4.5e-6 * (1 + math.cos(math.pi * (t - 10) / 290)) for t in range(1, 51)]
    return sum(lr[:steps])


def test_lower_top_keeps_block_scales():
    """MXFP4 blocks whose maximum is the top code sit on the scale-rounding boundary, so a growing maximum doubles the
    block scale. Release, 50 warmup-cosine steps, fixed sign: 40% of scales change on grid, 26% with the offset,
    2.95% with the maxima lowered."""
    master = _release_master((512, 1024), MXFP4)
    update = _warmup_cosine_total_lr() * torch.randn(master.shape, generator=torch.Generator().manual_seed(4)).sign()

    def scales_changed(m):
        before = requantize(m, MXFP4)[1]
        return float((requantize((m.float() + update).to(torch.bfloat16), MXFP4)[1] != before).float().mean())

    stalled = scales_changed(master)
    offset = scales_changed(offset_off_grid(master, MXFP4, lower_top=False))
    lowered = scales_changed(offset_off_grid(master, MXFP4))
    assert stalled > 0.25
    assert lowered < 0.05 and lowered < offset / 3


def test_offset_is_shard_and_stack_invariant():
    """Equal on every DP replica, for any 128-row shard and for an expert stack."""
    stack = torch.stack([_release_master((256, 256), E4M3_128_UE8M0, seed=s) for s in range(3)])
    whole = offset_off_grid(stack, E4M3_128_UE8M0)
    assert torch.equal(whole, offset_off_grid(stack, E4M3_128_UE8M0))
    for e in range(3):
        assert torch.equal(whole[e], offset_off_grid(stack[e], E4M3_128_UE8M0))
        assert torch.equal(whole[e, 128:], offset_off_grid(stack[e, 128:], E4M3_128_UE8M0))


@pytest.mark.parametrize(
    ("grid", "error", "refusal"),
    [
        (NVFP4, NotImplementedError, "global amax frozen"),
        (SyncGrid("e5m2", (128, 128), "ue8m0"), ValueError, "no off-grid offset"),
        (SyncGrid("e4m3", (1, 1), "per_tensor"), ValueError, "no off-grid offset"),
        (SyncGrid("e4m3", (64, 64), "ue8m0"), ValueError, "no off-grid offset"),
    ],
)
def test_unsupported_grids_are_refused(grid, error, refusal):
    with pytest.raises(error, match=refusal):
        offset_off_grid(torch.zeros(128, 128, dtype=torch.bfloat16), grid)


def test_mxfp4_needs_whole_groups():
    with pytest.raises(ValueError, match="MXFP4 groups"):
        offset_off_grid(torch.zeros(4, 48, dtype=torch.bfloat16), MXFP4)


def test_only_the_weight_sync_grids_are_pinned(monkeypatch):
    monkeypatch.setattr(off_grid, "PINNED_GRIDS", PINNED_GRIDS)
    assert PINNED_GRIDS == (SYNC_GRID, E4M3_32_UE8M0, MXFP4)
    for grid in PINNED_GRIDS:
        offset_off_grid(torch.zeros(128, 128, dtype=torch.bfloat16), grid)
    for grid in GRIDS:
        if grid not in PINNED_GRIDS:
            with pytest.raises(NotImplementedError, match=r"\[top\].*not pinned"):
                offset_off_grid(torch.zeros(128, 128, dtype=torch.bfloat16), grid)
