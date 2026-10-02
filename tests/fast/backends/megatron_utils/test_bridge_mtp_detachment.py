"""CPU regression tests for MTP detachment on bridge-built providers."""

import argparse
import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture(scope="module")
def apply_bridge_runtime_config() -> Callable:
    # Execute the production helper without importing GPU-only Megatron modules.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_bridge_runtime_config"
    )
    namespace = {"argparse": argparse, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_bridge_runtime_config"]


@pytest.fixture
def runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        gradient_accumulation_fusion=False,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=[],
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe=None,
        attention_backend="auto",
        moe_token_dispatcher_type="alltoall",
    )


@pytest.mark.parametrize(
    ("enabled", "initial_detach", "expected_detach"),
    [
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (False, True, True),
        (None, False, False),
        (None, True, True),
    ],
)
def test_bridge_mtp_detachment(
    apply_bridge_runtime_config: Callable,
    runtime_args: argparse.Namespace,
    enabled: bool | None,
    initial_detach: bool,
    expected_detach: bool,
) -> None:
    # A missing flag covers callers that only register Megatron's arguments.
    if enabled is not None:
        runtime_args.enable_mtp_training = enabled
    provider = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=initial_detach)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_detach_heads is expected_detach
    assert provider.mtp_num_layers == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_bridge_gelu_fusion(apply_bridge_runtime_config, runtime_args, enabled, gated):
    runtime_args.bias_gelu_fusion = enabled
    runtime_args.bias_swiglu_fusion = not enabled
    provider = SimpleNamespace(
        activation_func=torch.nn.functional.gelu, gated_linear_unit=gated, bias_activation_fusion=not enabled
    )
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.bias_activation_fusion is enabled
    assert provider.activation_func is torch.nn.functional.gelu
    assert provider.gated_linear_unit is gated


@pytest.mark.parametrize("enabled", [False, True])
def test_bridge_swiglu_fusion_uses_provider_activation(apply_bridge_runtime_config, runtime_args, enabled):
    runtime_args.swiglu = False  # Megatron CLI default does not describe the HF model.
    runtime_args.bias_swiglu_fusion = enabled
    runtime_args.bias_gelu_fusion = not enabled
    provider = SimpleNamespace(
        activation_func=torch.nn.functional.silu, gated_linear_unit=True, bias_activation_fusion=not enabled
    )
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.bias_activation_fusion is enabled
    assert provider.activation_func is torch.nn.functional.silu
    assert provider.gated_linear_unit is True


@pytest.mark.parametrize("activation,gated", [(torch.nn.functional.relu, False), (torch.nn.functional.silu, False)])
@pytest.mark.parametrize("initial", [False, True])
def test_bridge_preserves_other_activations(apply_bridge_runtime_config, runtime_args, activation, gated, initial):
    runtime_args.bias_swiglu_fusion = not initial
    runtime_args.bias_gelu_fusion = not initial
    provider = SimpleNamespace(activation_func=activation, gated_linear_unit=gated, bias_activation_fusion=initial)
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.bias_activation_fusion is initial
    assert provider.activation_func is activation
    assert provider.gated_linear_unit is gated


@pytest.mark.parametrize("activation,gated", [(torch.nn.functional.silu, True), (torch.nn.functional.gelu, False)])
@pytest.mark.parametrize("initial", [False, True])
def test_bridge_missing_activation_fusion_flag(apply_bridge_runtime_config, runtime_args, activation, gated, initial):
    provider = SimpleNamespace(activation_func=activation, gated_linear_unit=gated, bias_activation_fusion=initial)
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.bias_activation_fusion is initial
