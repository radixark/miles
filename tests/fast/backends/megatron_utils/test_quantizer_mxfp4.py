import torch

from miles.utils.mxfp4 import dequantize_mxfp4, project_mxfp4, quantize_mxfp4


def test_dequantize_mxfp4_decodes_nibbles_and_e8m0_scales() -> None:
    """Golden values pin the wire convention a pack/unpack round trip cannot catch."""
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
    scales = torch.tensor([127, 128], dtype=torch.uint8)
    expected = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 4.0, 6.0, 8.0, 12.0]],
        dtype=torch.bfloat16,
    )

    actual = dequantize_mxfp4(packed, scales, group_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_quantize_mxfp4_inverts_the_dequantizer() -> None:
    """Values read from an MXFP4 checkpoint re-encode to the same values, whatever scale the block was stored with."""
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8)
    scale = torch.randint(96, 144, (16, 4), dtype=torch.uint8)
    weight = dequantize_mxfp4(packed, scale, group_size=32)

    actual = dequantize_mxfp4(*quantize_mxfp4(weight, group_size=32), group_size=32)

    torch.testing.assert_close(actual, weight, rtol=0, atol=0)


def test_quantize_mxfp4_keeps_the_scale_when_a_top_code_block_max_grows() -> None:
    """A trained weight drifting just above a block's top code (6 * 2^s) must not coarsen the whole block."""
    grid = torch.tensor([6.0, -0.5, 1.5, 3.0, 0.0, -4.0, 1.0, 2.0] * 4) * 2.0**-7
    weight = grid.to(torch.bfloat16).reshape(1, 32)
    drifted = weight.clone()
    drifted[0, 0] = (drifted[0, 0].view(torch.int16) + 1).view(torch.bfloat16)  # the next BF16 value up

    packed, scale = quantize_mxfp4(drifted, group_size=32)

    assert scale.item() == 127 - 7
    torch.testing.assert_close(dequantize_mxfp4(packed, scale, group_size=32), weight, rtol=0, atol=0)


def test_quantize_mxfp4_keeps_the_scale_when_a_power_of_two_block_max_shrinks() -> None:
    """A block max on 4 * 2^s drifting just below it must not halve the scale and saturate at 3 * 2^s."""
    grid = torch.tensor([4.0, -0.5, 1.5, 3.0, 0.0, -2.0, 1.0, 2.0] * 4) * 2.0**-7
    weight = grid.to(torch.bfloat16).reshape(1, 32)
    drifted = weight.clone()
    drifted[0, 0] = (drifted[0, 0].view(torch.int16) - 1).view(torch.bfloat16)  # the next BF16 value down

    packed, scale = quantize_mxfp4(drifted, group_size=32)

    assert scale.item() == 127 - 7
    torch.testing.assert_close(dequantize_mxfp4(packed, scale, group_size=32), weight, rtol=0, atol=0)


def test_quantize_mxfp4_steps_the_scale_up_halfway_between_the_grid_maxima() -> None:
    """The scale moves from 2^s to 2^(s+1) at amax = 7 * 2^s, between the grid maxima 6 * 2^s and 8 * 2^s."""
    at = torch.full((1, 32), 7.0 * 2.0**-7).to(torch.bfloat16)
    below = (at.view(torch.int16) - 1).view(torch.bfloat16)  # the largest BF16 value under 7 * 2^s

    assert quantize_mxfp4(below, group_size=32)[1].item() == 127 - 7
    assert quantize_mxfp4(at, group_size=32)[1].item() == 127 - 6


def test_project_mxfp4_is_idempotent() -> None:
    """Projected weights re-encode to the same bits, so a sync of trainer weights is exact."""
    torch.manual_seed(0)
    weight = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)

    projected = project_mxfp4(weight, group_size=32)

    assert torch.equal(project_mxfp4(projected, group_size=32).view(torch.int16), projected.view(torch.int16))
