from types import SimpleNamespace
from unittest.mock import patch

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def _runtime_args(**overrides):
    values = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 8,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": False,
        "context_parallel_size": 1,
        "calculate_per_token_loss": False,
        "variable_seq_lengths": True,
        "attention_softmax_in_fp32": True,
        "gradient_accumulation_fusion": True,
        "fp32_residual_connection": False,
        "deterministic_mode": True,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": None,
        "cpu_offloading_num_layers": 0,
        "distribute_saved_activations": False,
        "tp_comm_overlap": False,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "attention_backend": "auto",
        "dsa_kernel_backend": "cudnn",
        "mtp_num_layers": None,
        "dsv4_mxfp4_qat": False,
        "moe_token_dispatcher_type": "alltoall",
        "decoder_first_pipeline_num_layers": 4,
        "decoder_last_pipeline_num_layers": 3,
        "moe_router_bias_update_rate": None,
        "moe_aux_loss_coeff": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider():
    return SimpleNamespace(
        dsa_kernel_backend="none",
        mtp_num_layers=1,
        experimental_attention_variant="dsv4_hybrid",
    )


def test_bridge_runtime_preserves_native_variant_and_disables_unrequested_mtp():
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args())

    assert provider.experimental_attention_variant == "dsv4_hybrid"
    assert provider.dsa_kernel_backend == "cudnn"
    assert provider.mtp_num_layers is None
    assert provider.dsv4_mxfp4_qat is False


def test_bridge_runtime_resolves_native_dsv4_kernel_default():
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args(dsa_kernel_backend=None))

    assert provider.dsa_kernel_backend == "cudnn"


def test_bridge_runtime_installs_mxfp4_qat_when_enabled():
    provider = _provider()

    with patch("miles_plugins.models.deepseek_v4.ops.mxfp4_qat.install_dsv4_mxfp4_qat") as install_qat:
        _apply_bridge_runtime_config(provider, _runtime_args(dsv4_mxfp4_qat=True))

    install_qat.assert_called_once_with()
    assert provider.dsv4_mxfp4_qat is True
