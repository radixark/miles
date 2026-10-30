import torch

from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, mxfp4_quantize


E8M0_BIAS = 127


def _dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 0x0F, (codes >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(E2M1_VALUES + tuple(-value for value in E2M1_VALUES), device=packed.device)
    values = table[nibbles.long()].reshape(-1, MXFP4_GROUP_SIZE)
    scale_factor = torch.exp2(scale.view(torch.uint8).float() - E8M0_BIAS).reshape(-1, 1)
    return (values * scale_factor).reshape(logical_shape)


def test_quantize_is_exact_on_representable_values():
    row = torch.tensor(E2M1_VALUES).repeat(MXFP4_GROUP_SIZE // len(E2M1_VALUES))
    weight = torch.stack([row, -row])

    packed, scale = mxfp4_quantize(weight)

    assert torch.equal(_dequantize_mxfp4(packed, scale), weight)


def test_quantize_round_trips_represented_values():
    torch.manual_seed(0)
    weight = torch.randn(64, 4 * MXFP4_GROUP_SIZE) * 0.05

    packed, scale = mxfp4_quantize(weight)
    decoded = _dequantize_mxfp4(packed, scale)

    repacked, rescale = mxfp4_quantize(decoded)
    assert torch.equal(_dequantize_mxfp4(repacked, rescale), decoded)


def test_quantize_keeps_error_within_the_grid_spacing():
    torch.manual_seed(0)
    weight = torch.randn(32, 2 * MXFP4_GROUP_SIZE)

    packed, scale = mxfp4_quantize(weight)
    decoded = _dequantize_mxfp4(packed, scale)

    blocks = weight.reshape(-1, MXFP4_GROUP_SIZE)
    exponent = scale.view(torch.uint8).float().reshape(-1, 1) - E8M0_BIAS
    tolerance = torch.exp2(exponent)
    assert ((decoded.reshape(-1, MXFP4_GROUP_SIZE) - blocks).abs() <= tolerance).all()


def test_quantize_emits_the_packed_layout():
    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 1.0
    weight[0, 1] = 6.0

    packed, scale = mxfp4_quantize(weight)

    assert packed.shape == (1, MXFP4_GROUP_SIZE // 2)
    assert scale.shape == (1, 1)
    assert packed.view(torch.uint8)[0, 0].item() == (7 << 4) | 2


def test_quantize_emits_scales_that_carry_their_value():
    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 48.0

    _, scale = mxfp4_quantize(weight)

    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.float().item() == 8.0
    assert scale.view(torch.uint8).item() == 3 + E8M0_BIAS


def test_quantize_encodes_zero_blocks_without_nan():
    packed, scale = mxfp4_quantize(torch.zeros(2, MXFP4_GROUP_SIZE))

    assert torch.equal(_dequantize_mxfp4(packed, scale), torch.zeros(2, MXFP4_GROUP_SIZE))
