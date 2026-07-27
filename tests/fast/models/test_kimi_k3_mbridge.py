from types import SimpleNamespace

import pytest
import torch

from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge


@pytest.mark.parametrize("name", ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias"))
def test_kda_state_stays_fp32_against_the_bridge_dtype(name):
    """The KDA short-conv weights and the recurrence's A_log / dt_bias are fp32
    parameters inside a bf16 model. ``_weight_to_mcore_format`` overrides the
    bridge dtype for exactly these; losing the override silently downcasts the
    delta-rule state during HF -> torch_dist conversion, which changes rollout
    numerics without any load-time error. Contrast the default path, which must
    still follow the bridge dtype.
    """
    bridge = object.__new__(KimiK3Bridge)
    bridge.dtype = torch.bfloat16
    bridge.config = SimpleNamespace(kimi_linear_num_heads=1)
    weight = torch.tensor([0.1234567], dtype=torch.float32)

    converted = bridge._weight_to_mcore_format(f"decoder.layers.0.self_attention.{name}", [weight])

    assert converted.dtype == torch.float32
    torch.testing.assert_close(converted, weight, rtol=0, atol=0)

    bridge.make_vocab_size_divisible_by = None
    assert (
        bridge._weight_to_mcore_format("decoder.layers.0.self_attention.q_proj.weight", [weight]).dtype
        == torch.bfloat16
    )
