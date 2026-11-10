import dataclasses
from types import SimpleNamespace

import pytest

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
        "mtp_detach_heads": True,
        "moe_token_dispatcher_type": "alltoall",
        "decoder_first_pipeline_num_layers": 4,
        "decoder_last_pipeline_num_layers": 3,
        "moe_router_bias_update_rate": None,
        "moe_aux_loss_coeff": None,
    }
    values.update(overrides)
    return SimpleNamespace(
        backend=SimpleNamespace(**values),
        enable_mtp_training=False,
        dsa_attention_backend=values.get("dsa_attention_backend"),
        routed_expert_qat=values.get("routed_expert_qat"),
    )


@dataclasses.dataclass
class _Provider:
    dsa_kernel_backend: str = "none"
    mtp_num_layers: int | None = 1
    experimental_attention_variant: str = "dsv4_hybrid"
    moe_expert_fake_quant: str | None = None
    moe_grouped_gemm: bool = True
    use_transformer_engine_op_fuser: bool = False
    delay_wgrad_compute: bool = False
    overlap_dispatch_backward_with_experts_wgrad: bool = False
    cuda_graph_impl: str = "none"
    enable_cuda_graph: bool = False
    external_cuda_graph: bool = False
    fp4: str | None = None


@dataclasses.dataclass
class _ProviderWithoutFakeQuantField:
    dsa_kernel_backend: str = "none"
    mtp_num_layers: int | None = 1
    experimental_attention_variant: str = "dsv4_hybrid"
    moe_grouped_gemm: bool = True
    fp4: str | None = None


def _provider():
    return _Provider()


def test_bridge_runtime_preserves_native_variant_and_disables_unrequested_mtp():
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args())

    assert provider.experimental_attention_variant == "dsv4_hybrid"
    assert provider.dsa_kernel_backend == "cudnn"
    assert provider.mtp_num_layers is None
    assert provider.moe_expert_fake_quant is None


@pytest.mark.parametrize("backend", [None, "none", "tilelang", "cudnn"])
def test_bridge_runtime_resolves_native_dsv4_kernel_default(backend):
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args(dsa_kernel_backend=backend, dsa_attention_backend="tilelang"))

    assert provider.dsa_kernel_backend == (backend or "cudnn")


def test_bridge_runtime_sets_routed_expert_fake_quant_when_enabled():
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args(routed_expert_qat="mxfp4"))

    assert provider.moe_expert_fake_quant == "mxfp4"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("moe_grouped_gemm", False, "grouped-GEMM routed experts"),
        ("use_transformer_engine_op_fuser", True, "op fuser"),
        ("cuda_graph_impl", "local", "CUDA graphs"),
        ("cuda_graph_impl", "transformer_engine", "CUDA graphs"),
        ("cuda_graph_impl", "full_iteration", "CUDA graphs"),
        ("enable_cuda_graph", True, "CUDA graphs"),
        ("external_cuda_graph", True, "CUDA graphs"),
    ],
)
def test_bridge_runtime_rejects_fake_quant_that_would_bypass_the_hook(field, value, message):
    provider = _provider()
    setattr(provider, field, value)

    with pytest.raises(ValueError, match=message):
        _apply_bridge_runtime_config(provider, _runtime_args(routed_expert_qat="mxfp4"))


@pytest.mark.parametrize("field", ["delay_wgrad_compute", "overlap_dispatch_backward_with_experts_wgrad"])
def test_bridge_runtime_accepts_fake_quant_with_delayed_weight_gradients(field):
    """Delayed wgrad reads the saved activations and output gradients, not the rounded weights."""
    provider = _provider()
    setattr(provider, field, True)

    _apply_bridge_runtime_config(provider, _runtime_args(routed_expert_qat="mxfp4"))

    assert provider.moe_expert_fake_quant == "mxfp4"


def test_bridge_runtime_rejects_fake_quant_without_a_first_microbatch_signal():
    with pytest.raises(ValueError, match="FP8 or FP4 training recipe"):
        _apply_bridge_runtime_config(_provider(), _runtime_args(routed_expert_qat="mxfp4", fp8=None))


def test_bridge_runtime_rejects_fake_quant_on_a_megatron_without_the_field():
    with pytest.raises(ValueError, match="no moe_expert_fake_quant field"):
        _apply_bridge_runtime_config(_ProviderWithoutFakeQuantField(), _runtime_args(routed_expert_qat="mxfp4"))


@pytest.mark.parametrize("flag", ["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "OPEN_TRAINING_INT4_FAKE_QAT_FLAG"])
def test_bridge_runtime_rejects_combining_fake_quant_formats(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")

    with pytest.raises(ValueError, match="cannot be combined"):
        _apply_bridge_runtime_config(_provider(), _runtime_args(routed_expert_qat="mxfp4"))
