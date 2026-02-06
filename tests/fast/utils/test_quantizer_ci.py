"""Tests for the prefix-matching ignore rule added to quantize_params_compressed_tensors.

The change adds `name.startswith(r)` to the ignore matching logic, so rules like
"model.layers.0.self_attn" now ignore all weights under that prefix.
"""

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    quantize_params_compressed_tensors,
)

CONFIG = {
    "config_groups": {"group_0": {"weights": {"group_size": 128, "symmetric": True}}},
    "ignore": [],  # overridden per test
}


def _quantize_names(name, ignore_rules):
    """Run quantization on a single 2D weight and return output names."""
    config = {**CONFIG, "ignore": ignore_rules}
    results = quantize_params_compressed_tensors(
        [(name, torch.randn(256, 256))], config
    )
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
            ("model.layers.0.self_attn.q_proj.weight", torch.randn(256, 256)),
            ("model.layers.0.self_attn.k_proj.weight", torch.randn(256, 256)),
            ("model.layers.0.mlp.gate_proj.weight", torch.randn(256, 256)),
        ]
        result_names = [r[0] for r in quantize_params_compressed_tensors(params, config)]

        # self_attn params ignored (passed through)
        assert "model.layers.0.self_attn.q_proj.weight" in result_names
        assert "model.layers.0.self_attn.k_proj.weight" in result_names
        # mlp param quantized
        assert "model.layers.0.mlp.gate_proj.weight_packed" in result_names
