import torch
import torch.nn as nn

from miles_plugins.models.kimi_k3.checkpoint import dequantize_mxfp4
from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, attn_res_aggregate, situ_and_mul


def test_situ_and_mul_matches_fp32_reference() -> None:
    x = torch.tensor(
        [[-40.0, -3.0, 2.0, 30.0, -50.0, -2.0, 4.0, 60.0]],
        dtype=torch.bfloat16,
    )
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * 25.0 * torch.tanh(linear / 25.0)).to(x.dtype)

    torch.testing.assert_close(situ_and_mul(x), expected, rtol=0, atol=0)


def test_attn_res_aggregate_matches_reference() -> None:
    torch.manual_seed(123)
    prefix_sum = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    block_residual = torch.randn(3, 2, 4, 8, dtype=torch.bfloat16)
    score_norm = KimiRMSNorm(8, eps=1e-5)
    score_proj = nn.Linear(8, 1, bias=False)
    output_norm = KimiRMSNorm(8, eps=1e-5)

    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    normalized = rows.float() * torch.rsqrt(rows.float().square().mean(-1, keepdim=True) + 1e-5)
    scores = normalized @ (score_norm.weight.float() * score_proj.weight[0].float())
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows.float()).sum(-2)
    expected = output_norm(mixed.to(rows.dtype))

    actual = attn_res_aggregate(
        prefix_sum,
        block_residual,
        score_proj,
        score_norm,
        output_norm,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dequantize_mxfp4_decodes_nibbles_and_e8m0_scales() -> None:
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
    scales = torch.tensor([127, 128], dtype=torch.uint8)
    expected = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 4.0, 6.0, 8.0, 12.0]],
        dtype=torch.bfloat16,
    )

    actual = dequantize_mxfp4(packed, scales, group_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
