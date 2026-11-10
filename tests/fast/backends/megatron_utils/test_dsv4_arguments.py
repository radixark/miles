from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.utils.args.configs.custom_megatron_plugins import Dsv4MegatronPluginsConfig
from miles.utils.arguments import validate_routed_expert_quantization_args
from miles_plugins.models.deepseek_v4.arguments import DSV4_SPEC_MODULE, is_dsv4_model, normalize_dsv4_args


def _parse(*argv: str) -> Namespace:
    parser = ArgumentParser()
    Dsv4MegatronPluginsConfig.add_arguments(parser=parser)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--dsa-kernel-backend", default=None)
    parser.add_argument("--spec", nargs="*", default=[DSV4_SPEC_MODULE, "get_dsv4_spec"])
    return parser.parse_args(argv)


def test_only_the_dsv4_spec_triggers_normalization():
    assert is_dsv4_model(_parse())

    other = _parse()
    other.spec = ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]
    assert not is_dsv4_model(other)

    unspecced = _parse()
    unspecced.spec = None
    assert not is_dsv4_model(unspecced)


def test_impl_selects_the_attention_variant():
    default = _parse()
    normalize_dsv4_args(default)
    assert default.experimental_attention_variant == "dsv4_hybrid"

    miles = _parse("--dsv4-impl", "miles")
    normalize_dsv4_args(miles)
    assert miles.experimental_attention_variant == "dsv4"

    megatron = _parse("--dsv4-impl", "megatron")
    normalize_dsv4_args(megatron)
    assert megatron.experimental_attention_variant == "dsv4_hybrid"
    assert megatron.enable_hyper_connections


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--dsv4-impl", "megatron", "--tensor-model-parallel-size", "8"), "tensor-model-parallel-size 1"),
        (("--dsv4-impl", "megatron", "--dsa-kernel-backend", "tilelang"), "does not support"),
        (("--dsv4-impl", "miles", "--dsa-kernel-backend", "cudnn"), "ignores cuDNN"),
    ],
    ids=["megatron-needs-tp1", "megatron-rejects-tilelang", "miles-rejects-cudnn"],
)
def test_unsupported_combinations_fail_at_parse_time(argv, message):
    """Megatron would only assert deep inside config post-init, or silently mis-run."""
    with pytest.raises(ValueError, match=message):
        normalize_dsv4_args(_parse(*argv))


def _quantization_args(**overrides):
    return Namespace(
        **(
            {
                "routed_expert_qat": None,
                "megatron_to_hf_mode": "bridge",
                "rollout_routed_expert_format": None,
                "bf16": True,
                "fp16": False,
                "use_precision_aware_optimizer": False,
                "main_params_dtype": torch.float32,
                "fp8_param_gather": False,
                "lora_rank": 0,
                "lora_adapter_path": None,
                "custom_model_provider_path": None,
                "cuda_graph_impl": "none",
            }
            | overrides
        )
    )


def _fp8_checkpoint_config(model_type="deepseek_v4"):
    return SimpleNamespace(model_type=model_type, quantization_config={"quant_method": "fp8"})


def test_routed_expert_quantization_flags_reach_a_consumer():
    fp8 = _fp8_checkpoint_config()

    validate_routed_expert_quantization_args(
        _quantization_args(routed_expert_qat="mxfp4", rollout_routed_expert_format="mxfp4"), fp8
    )

    with pytest.raises(ValueError, match="Megatron-Bridge provider"):
        validate_routed_expert_quantization_args(
            _quantization_args(routed_expert_qat="mxfp4", megatron_to_hf_mode="raw"), fp8
        )
    with pytest.raises(ValueError, match="block-wise FP8 rollout checkpoint"):
        validate_routed_expert_quantization_args(
            _quantization_args(rollout_routed_expert_format="mxfp8"),
            SimpleNamespace(model_type="deepseek_v4", quantization_config={"quant_method": "mxfp8"}),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"custom_model_provider_path": "pkg.model_provider"}, "--custom-model-provider-path replaces"),
        ({"cuda_graph_impl": "local"}, "CUDA graphs"),
        ({"cuda_graph_impl": "transformer_engine"}, "CUDA graphs"),
        ({"cuda_graph_impl": "full_iteration"}, "CUDA graphs"),
        ({"enable_cuda_graph": True}, "CUDA graphs"),
        ({"external_cuda_graph": True}, "CUDA graphs"),
    ],
)
def test_routed_expert_qat_rejects_paths_that_would_skip_the_rounding(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_routed_expert_quantization_args(
            _quantization_args(routed_expert_qat="mxfp4", **overrides), _fp8_checkpoint_config()
        )


def test_routed_expert_qat_accepts_fp32_masters_of_a_precision_aware_optimizer():
    validate_routed_expert_quantization_args(
        _quantization_args(routed_expert_qat="mxfp4", use_precision_aware_optimizer=True), _fp8_checkpoint_config()
    )


@pytest.mark.parametrize(
    ("overrides", "model_type", "message"),
    [
        ({}, "kimi_k3", "supports model types"),
        ({"bf16": False}, "deepseek_v4", "fp32 master weights"),
        (
            {"use_precision_aware_optimizer": True, "main_params_dtype": torch.float16},
            "deepseek_v4",
            "--main-params-dtype fp32",
        ),
        ({"fp8_param_gather": True}, "deepseek_v4", "FP8 primary weights"),
        ({"lora_rank": 8}, "deepseek_v4", "LoRA"),
    ],
    ids=["unverified-model", "no-mixed-precision", "fp16-masters", "fp8-primary-weights", "lora"],
)
def test_routed_expert_qat_rejects_unverified_models_and_shared_master_weights(overrides, model_type, message):
    """Rounding in place is straight-through only when the optimizer keeps separate fp32 master weights."""
    with pytest.raises(ValueError, match=message):
        validate_routed_expert_quantization_args(
            _quantization_args(routed_expert_qat="mxfp4", **overrides), _fp8_checkpoint_config(model_type)
        )
