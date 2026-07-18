"""Tests for the prefix-matching ignore rule added to quantize_params_compressed_tensors.

The change adds `name.startswith(r)` to the ignore matching logic, so rules like
"model.layers.0.self_attn" now ignore all weights under that prefix.
"""

from tests.ci.ci_register import register_cuda_ci

# The quantizer hardcodes `device="cuda"` throughout; this test drives it with
# real CUDA tensors to exercise the ignore-rule name-matching path. Fast enough
# for the GPU fast suite; only needs 1 GPU.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])


import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    quantize_mxfp4,
    quantize_params_compressed_tensors,
)
from miles_plugins.models.kimi_k3.checkpoint import dequantize_mxfp4

CONFIG = {
    "format": "int-quantized",
    "config_groups": {"group_0": {"weights": {"group_size": 128, "symmetric": True}}},
    "ignore": [],  # overridden per test
}


def _quantize_names(name, ignore_rules):
    """Run quantization on a single 2D weight and return output names."""
    config = {**CONFIG, "ignore": ignore_rules}
    results = quantize_params_compressed_tensors([(name, torch.randn(256, 256, device="cuda"))], config)
    return [r[0] for r in results]


def _is_ignored(name, ignore_rules):
    """Check if a weight name is ignored (returned as-is, not quantized)."""
    names = _quantize_names(name, ignore_rules)
    return name in names and f"{name}_packed" not in names


class TestIgnoreRulePrefixMatching:
    """Tests for the new prefix-matching ignore rule (name.startswith(r))."""

    @pytest.mark.parametrize(
        "rule,name,expected_ignored",
        [
            # exact match (pre-existing)
            ("model.layer.weight", "model.layer.weight", True),
            # regex match (pre-existing)
            ("re:.*embed.*", "model.embed_tokens.weight", True),
            # prefix match (NEW)
            ("model.layers.0.self_attn", "model.layers.0.self_attn.q_proj.weight", True),
            ("model.layers.", "model.layers.5.attn.weight", True),
            ("model.embed", "model.embed_tokens.weight", True),
            # non-matching
            ("model.layers.1", "model.layers.0.attn.weight", False),
            ("other.prefix", "model.layers.0.attn.weight", False),
        ],
    )
    def test_ignore_rule_matching(self, rule, name, expected_ignored):
        assert _is_ignored(name, [rule]) == expected_ignored

    def test_prefix_selectively_ignores(self):
        """Prefix rule ignores matching params while others get quantized."""
        config = {**CONFIG, "ignore": ["model.layers.0.self_attn"]}
        params = [
            ("model.layers.0.self_attn.q_proj.weight", torch.randn(256, 256, device="cuda")),
            ("model.layers.0.self_attn.k_proj.weight", torch.randn(256, 256, device="cuda")),
            ("model.layers.0.mlp.gate_proj.weight", torch.randn(256, 256, device="cuda")),
        ]
        result_names = [r[0] for r in quantize_params_compressed_tensors(params, config)]

        # self_attn params ignored (passed through)
        assert "model.layers.0.self_attn.q_proj.weight" in result_names
        assert "model.layers.0.self_attn.k_proj.weight" in result_names
        # mlp param quantized
        assert "model.layers.0.mlp.gate_proj.weight_packed" in result_names


def test_mxfp4_round_trip_preserves_dequantized_values():
    group_size = 32
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
    scale = torch.randint(96, 144, (16, 4), dtype=torch.uint8, device="cuda")
    weight = dequantize_mxfp4(packed, scale, group_size)

    actual_packed, actual_scale = quantize_mxfp4(weight, group_size)
    actual = dequantize_mxfp4(actual_packed, actual_scale, group_size)

    torch.testing.assert_close(actual, weight, rtol=0, atol=0)


def test_mxfp4_compressed_tensor_names_and_dtypes():
    config = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "group_size": 32,
                    "symmetric": True,
                    "type": "float",
                    "num_bits": 4,
                    "scale_dtype": "torch.uint8",
                }
            }
        },
        "ignore": [],
    }
    results = quantize_params_compressed_tensors(
        [("model.layers.0.experts.0.w1.weight", torch.randn(64, 64, device="cuda"))],
        config,
    )

    assert [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in results] == [
        ("model.layers.0.experts.0.w1.weight_packed", torch.uint8, (64, 32)),
        ("model.layers.0.experts.0.w1.weight_scale", torch.uint8, (64, 2)),
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
