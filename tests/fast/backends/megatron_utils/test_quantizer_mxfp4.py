import torch

from miles.utils.mxfp4 import dequantize_mxfp4


def test_dequantize_mxfp4_decodes_nibbles_and_e8m0_scales() -> None:
    """Hand-computed golden values for the MXFP4 wire format: low nibble first,
    bit 3 is the sign, magnitude indexes the e2m1 table (0, .5, 1, 1.5, 2, 3, 4,
    6), and the uint8 scale is a base-2 exponent biased by 127. The round-trip
    test in test_quantizer_ci.py cannot catch a consistently-wrong convention
    here -- only fixed expected values can.
    """
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
    scales = torch.tensor([127, 128], dtype=torch.uint8)
    expected = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 4.0, 6.0, 8.0, 12.0]],
        dtype=torch.bfloat16,
    )

    actual = dequantize_mxfp4(packed, scales, group_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
