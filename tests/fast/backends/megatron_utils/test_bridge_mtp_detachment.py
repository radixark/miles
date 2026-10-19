"""CPU regression tests for MTP detachment on bridge-built providers."""

import argparse
import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    namespace = {"argparse": argparse}
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
        mtp_num_layers=1,
        mtp_detach_heads=True,
        moe_token_dispatcher_type="alltoall",
    )


@pytest.mark.parametrize(
    ("detach", "initial_detach"),
    [(True, False), (True, True), (False, False), (False, True)],
)
def test_bridge_mtp_detachment(
    apply_bridge_runtime_config: Callable,
    runtime_args: argparse.Namespace,
    detach: bool,
    initial_detach: bool,
) -> None:
    """The provider takes --mtp-detach-heads from args, whatever the bridge's HF config set."""
    runtime_args.mtp_detach_heads = detach
    provider = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=initial_detach)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_detach_heads is detach
    assert provider.mtp_num_layers == 1


@pytest.mark.parametrize(
    ("mtp_args", "expected_layers"),
    [({}, 1), ({"mtp_num_layers": None}, None), ({"mtp_num_layers": 2}, 2)],
    ids=["absent", "disabled", "enabled"],
)
def test_bridge_mtp_runtime_depth(apply_bridge_runtime_config, runtime_args, mtp_args, expected_layers):
    del runtime_args.mtp_num_layers
    vars(runtime_args).update(mtp_args)
    provider = SimpleNamespace(mtp_num_layers=1)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_num_layers == expected_layers
