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
        "moe_router_bias_update_rate",
        "moe_aux_loss_coeff",
    }

    _apply_bridge_runtime_config(provider, args)

    assert {name: vars(provider)[name] for name in same_name_fields} == {
        name: vars(args.backend)[name] for name in same_name_fields
    }
    assert provider.num_layers_in_first_pipeline_stage == (5 if explicit_options else 11)
    assert provider.num_layers_in_last_pipeline_stage == (7 if explicit_options else 13)
    assert provider.moe_router_bias_update_rate == (0.125 if explicit_options else 0.5)
    assert provider.moe_aux_loss_coeff == (0.25 if explicit_options else 0.75)
    assert provider.dsa_kernel_backend == "cudnn"
