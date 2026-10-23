"""Routing of online-update quantization for rollout routed experts."""

from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.megatron_to_hf import processors

_EXPERT = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight7"
_SHARED_EXPERT = "module.module.decoder.layers.3.mlp.shared_experts.linear_fc1.weight"
_FP8_CONFIG = {"quant_method": "fp8", "weight_block_size": [128, 128]}


@pytest.fixture
def quantizers(monkeypatch):
    calls = []
    for name in ("quantize_params_fp8", "quantize_params_mxfp4", "quantize_routed_expert_params_mxfp8"):
        monkeypatch.setattr(processors, name, lambda *args, _name=name: calls.append(_name) or [])
    return calls


@pytest.mark.parametrize(
    ("expert_format", "expected"),
    [
        (None, "quantize_params_fp8"),
        ("mxfp8", "quantize_routed_expert_params_mxfp8"),
        ("mxfp4", "quantize_params_mxfp4"),
    ],
)
def test_routed_experts_follow_the_rollout_format(quantizers, expert_format, expected):
    args = SimpleNamespace(rollout_routed_expert_format=expert_format)

    processors.quantize_params(args, _EXPERT, [], _FP8_CONFIG)

    assert quantizers == [expected]


@pytest.mark.parametrize("expert_format", ["mxfp8", "mxfp4"])
def test_other_weights_stay_block_wise_fp8(quantizers, expert_format):
    args = SimpleNamespace(rollout_routed_expert_format=expert_format)

    processors.quantize_params(args, _SHARED_EXPERT, [], _FP8_CONFIG)

    assert quantizers == ["quantize_params_fp8"]


@pytest.mark.parametrize("expert_format", ["mxfp8", "mxfp4"])
def test_routed_experts_excluded_by_the_checkpoint_stay_on_the_fp8_path(quantizers, expert_format):
    args = SimpleNamespace(rollout_routed_expert_format=expert_format)
    config = _FP8_CONFIG | {"modules_to_not_convert": ["layers.3.ffn.experts"]}

    processors.quantize_params(args, _EXPERT, [("layers.3.ffn.experts.7.w1.weight", None)], config)

    assert quantizers == ["quantize_params_fp8"]
