from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

import sys

import pytest


def test_post_layernorm_flags_propagate_to_megatron(monkeypatch):
    pytest.importorskip("megatron.training.arguments")

    import torch
    from megatron.training.arguments import core_transformer_config_from_args

    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)

    argv = [
        "pytest",
        "--train-backend",
        "megatron",
        "--rollout-batch-size",
        "1",
        "--num-layers",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--post-self-attn-layernorm",
        "--post-mlp-layernorm",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = miles_arguments.parse_args()

    assert args.post_self_attn_layernorm is True
    assert args.post_mlp_layernorm is True

    if args.bf16:
        args.params_dtype = torch.bfloat16
    elif args.fp16:
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32

    # apply_rope_fusion requires TransformerEngine >= 1.4, which is GPU-only
    # and not installed on CPU CI. This test only validates post-layernorm flag
    # propagation, so disable the fused kernel to avoid TransformerConfig
    # __post_init__ validation failure.
    args.apply_rope_fusion = False

    config = core_transformer_config_from_args(args)

    assert config.post_self_attn_layernorm is True
    assert config.post_mlp_layernorm is True
