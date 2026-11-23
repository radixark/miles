"""DeepSeek-V4.1 32x32 block-FP8 masters: what the trainer refuses instead of installing a different program."""

import pytest
import torch

from miles_plugins.top.fp8_b32 import TrainableFP8B32Linear, dequantize_b32, quantize_b32, trainable_clamped_swiglu_mlp


def _checkpoint(n=64, k=96):
    """A release-shaped pair: what the UE8M0 32x32 quantizer writes for BF16 weights (|w| ~0.023, as V4.1's)."""
    g = torch.Generator().manual_seed(0)
    return quantize_b32((torch.randn(n, k, generator=g) * 0.023).to(torch.bfloat16))


def test_non_power_of_two_scales_are_refused():
    """Serving reads the scales as loaded; a non-UE8M0 scale would make the BF16 master inexact."""
    payload, scale = _checkpoint()
    with pytest.raises(ValueError, match="powers of two"):
        dequantize_b32(payload, scale * 1.5)


def test_a_128x128_scale_grid_is_refused():
    payload, _ = _checkpoint(n=128, k=128)
    with pytest.raises(ValueError, match="32x32 block grid"):
        dequantize_b32(payload, torch.ones(1, 1))


def test_an_external_weight_is_refused():
    layer = TrainableFP8B32Linear(*_checkpoint())
    with pytest.raises(NotImplementedError, match="external weight"):
        layer(torch.zeros(2, 96, dtype=torch.bfloat16), weight=torch.zeros(64, 96))


def test_the_shared_expert_refuses_another_clamp():
    fc1, fc2 = TrainableFP8B32Linear(*_checkpoint(n=128, k=96)), TrainableFP8B32Linear(*_checkpoint(n=96, k=64))
    with pytest.raises(NotImplementedError, match="clamps at 10.0"):
        trainable_clamped_swiglu_mlp(torch.zeros(2, 96, dtype=torch.bfloat16), fc1, fc2, None)


def test_the_shared_expert_refuses_a_128x128_projection():
    """A DS4 TrainableFP8Linear would run the 128-block program on one side only."""
    fc2 = TrainableFP8B32Linear(*_checkpoint(n=96, k=64))
    with pytest.raises(NotImplementedError, match="32x32 FP8 projections"):
        trainable_clamped_swiglu_mlp(torch.zeros(2, 96, dtype=torch.bfloat16), torch.nn.Linear(96, 128), fc2, 10.0)
