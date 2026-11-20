from types import SimpleNamespace

import pytest
from tests.fast.backends.megatron_utils.test_dsv4_bridge_runtime import _runtime_args

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


@pytest.mark.parametrize("explicit_options", [False, True])
def test_bridge_provider_receives_backend_runtime_options_and_preserves_unset_overrides(
    explicit_options: bool,
) -> None:
    """Bridge runtime fields use the backend while unset overrides retain checkpoint values."""
    args = _runtime_args(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=3,
        cpu_offloading=True,
        cpu_offloading_num_layers=2,
        recompute_num_layers=3,
        recompute_modules=["core_attn"],
        decoder_first_pipeline_num_layers=5 if explicit_options else None,
        decoder_last_pipeline_num_layers=7 if explicit_options else None,
        moe_router_bias_update_rate=0.125 if explicit_options else None,
        moe_aux_loss_coeff=0.25 if explicit_options else None,
        fp4="e2m1",
        fp4_param_gather=True,
        first_last_layers_bf16=True,
        num_layers_at_end_in_bf16=2,
    )
    provider = SimpleNamespace(
        experimental_attention_variant="dsv4_hybrid",
        num_layers_in_first_pipeline_stage=11,
        num_layers_in_last_pipeline_stage=13,
        moe_router_bias_update_rate=0.5,
        moe_aux_loss_coeff=0.75,
    )
    same_name_fields = set(vars(args.backend)) - {
        "dsa_kernel_backend",
        "decoder_first_pipeline_num_layers",
        "decoder_last_pipeline_num_layers",
        "pipeline_model_parallel_layout",
        "moe_router_bias_update_rate",
        "moe_aux_loss_coeff",
        # renamed on the way to the provider, as in core_transformer_config_from_args
        "fp8_param_gather",
        "fp4_param_gather",
        "te_precision_config_file",
    }

    _apply_bridge_runtime_config(provider, args)

    assert {name: vars(provider)[name] for name in same_name_fields} == {
        name: vars(args.backend)[name] for name in same_name_fields
    }
    assert provider.fp8_param is False
    assert provider.fp4_param is True
    assert provider.quant_recipe is None
    assert provider.num_layers_in_first_pipeline_stage == (5 if explicit_options else 11)
    assert provider.num_layers_in_last_pipeline_stage == (7 if explicit_options else 13)
    assert provider.moe_router_bias_update_rate == (0.125 if explicit_options else 0.5)
    assert provider.moe_aux_loss_coeff == (0.25 if explicit_options else 0.75)
    assert provider.dsa_kernel_backend == "cudnn"


def test_bridge_provider_parses_te_precision_config_file(tmp_path) -> None:
    """A bridge-built model must not ignore --te-precision-config-file."""
    from megatron.core.quantization.quant_config import MatchContext

    config_file = tmp_path / "te_precision.yaml"
    config_file.write_text(
        """
configs:
    nvfp4:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe:
            fp4_quantization_recipe: "nvfp4"
    bf16:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe: {}
matchers:
    routed_experts_fc1_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc1"
        config: "nvfp4"
    default_bf16:
        type: "glob"
        enabled: true
        pattern: "*"
        config: "bf16"
"""
    )
    provider = SimpleNamespace()

    _apply_bridge_runtime_config(provider, _runtime_args(te_precision_config_file=str(config_file)))

    def matched(module_path: str) -> str:
        return provider.quant_recipe.match(MatchContext(module_path=module_path, layer_number=0)).config_key

    assert matched("decoder.layers.0.mlp.experts.linear_fc1") == "nvfp4"
    assert matched("decoder.layers.0.self_attention.linear_qkv") == "bf16"


def test_bridge_provider_sets_every_precision_field() -> None:
    """A Megatron precision field must reach bridge-built models or be listed as having no command-line flag."""
    import dataclasses

    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles.utils.megatron_bridge_utils import BRIDGE_PRECISION_FIELDS

    no_command_line_flag = {
        "fp8_dot_product_attention",
        "fp8_multi_head_attention",
        "use_kitchen",
        "use_kitchen_attention",
        "kitchen_attention_backend",
    }
    precision_fields = {
        field.name
        for field in dataclasses.fields(TransformerConfig)
        if field.name.startswith(("fp8", "fp4", "kitchen", "use_kitchen"))
        or field.name
        in {"first_last_layers_bf16", "num_layers_at_start_in_bf16", "num_layers_at_end_in_bf16", "quant_recipe"}
    }

    assert precision_fields - no_command_line_flag == set(BRIDGE_PRECISION_FIELDS) | {"quant_recipe"}
