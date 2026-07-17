import pytest
import torch

from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge


@pytest.mark.parametrize("projection", ("q", "k", "v"))
def test_kda_conv_weights_remain_fp32(projection):
    bridge = object.__new__(KimiK3Bridge)
    bridge.dtype = torch.bfloat16
    weight = torch.tensor([0.1234567], dtype=torch.float32)

    converted = bridge._weight_to_mcore_format(f"decoder.layers.0.self_attention.{projection}_conv1d.weight", [weight])

    assert converted.dtype == torch.float32
    torch.testing.assert_close(converted, weight, rtol=0, atol=0)


def test_regular_weights_follow_bridge_dtype():
    bridge = object.__new__(KimiK3Bridge)
    bridge.dtype = torch.bfloat16
    bridge.make_vocab_size_divisible_by = None
    weight = torch.tensor([0.1234567], dtype=torch.float32)

    converted = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.q_proj.weight", [weight])

    assert converted.dtype == torch.bfloat16
