import torch

from miles.utils.mxfp4 import dequantize_mxfp4


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
