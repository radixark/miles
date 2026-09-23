"""`quantize_params_compressed_tensors` re-quantizes exactly the weights the checkpoint stores packed."""

from tests.ci.ci_register import register_cuda_ci

# The quantizer hardcodes `device="cuda"` throughout, so it runs on a GPU worker.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])


import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    quantize_params_compressed_tensors,
)
from miles.utils.mxfp4 import dequantize_mxfp4, quantize_mxfp4

CONFIG = {
    "format": "int-quantized",
    "config_groups": {"group_0": {"weights": {"group_size": 128, "symmetric": True}}},
}


def test_only_the_packed_basenames_are_quantized():
    """A 2-D BF16 weight the checkpoint keeps unpacked (router, residual projection) must pass through."""
    params = [
        ("model.layers.0.experts.0.w1.weight", torch.randn(256, 256, device="cuda")),
        ("model.layers.0.gate.weight", torch.randn(256, 256, device="cuda")),
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(256, 256, device="cuda")),
    ]
    result_names = [
        name for name, _ in quantize_params_compressed_tensors(params, CONFIG, {"model.layers.0.experts.0.w1"})
    ]

    assert "model.layers.0.experts.0.w1.weight_packed" in result_names
    assert "model.layers.0.experts.0.w1.weight" not in result_names
    assert "model.layers.0.gate.weight" in result_names
    assert "model.layers.0.self_attn.q_proj.weight" in result_names


def test_mxfp4_quantize_inverts_the_checkpoint_dequantizer():
    """``quantize_mxfp4`` must be the exact inverse of the dequantizer used to
    read the K3 checkpoint: any value that came out of an MXFP4 checkpoint has
    to re-encode to the same bits, or every weight sync degrades the rollout
    weights a little further. The magnitude thresholds, sign-bit position,
    nibble order and exponent bias all have to agree.

    Also pins the mxfp4 branch of ``quantize_params_compressed_tensors``, which
    emits only weight_packed/weight_scale -- the int-quantized branch's extra
    weight_shape/weight_zero_point tensors would be rejected by the receiver.
    """
    group_size = 32
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
    scale = torch.randint(96, 144, (16, 4), dtype=torch.uint8, device="cuda")
    weight = dequantize_mxfp4(packed, scale, group_size)

    actual_packed, actual_scale = quantize_mxfp4(weight, group_size)
    actual = dequantize_mxfp4(actual_packed, actual_scale, group_size)

    torch.testing.assert_close(actual, weight, rtol=0, atol=0)

    config = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "group_size": group_size,
                    "symmetric": True,
                    "type": "float",
                    "num_bits": 4,
                    "scale_dtype": "torch.uint8",
                }
            }
        },
    }
    results = quantize_params_compressed_tensors(
        [("model.layers.0.experts.0.w1.weight", torch.randn(64, 64, device="cuda"))],
        config,
        {"model.layers.0.experts.0.w1"},
    )

    assert [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in results] == [
        ("model.layers.0.experts.0.w1.weight_packed", torch.uint8, (64, 32)),
        ("model.layers.0.experts.0.w1.weight_scale", torch.uint8, (64, 2)),
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
