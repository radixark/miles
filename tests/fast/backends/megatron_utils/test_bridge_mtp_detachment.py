"""CPU regression tests for the MTP rule: a bridge builds only the MTP layers its trainer names, detached.

Megatron adds the MTP loss to every training forward of a model that has MTP layers, rolling the
labels from input_ids when labels is None. Undetached, it trains the policy on its own samples.
"""

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
        moe_token_dispatcher_type="alltoall",
    )


@pytest.mark.parametrize(
    ("named", "inherited", "expected"),
    [
        # A trainer that does not train MTP names 0, so the HF config's MTP layer is not built.
        (0, 1, None),
        # Checkpoint conversion leaves --mtp-num-layers unset and keeps the checkpoint's MTP layer.
        (None, 1, 1),
        (1, 1, 1),
        # The argument never adds an MTP layer the bridge does not build (e.g. GLM-5.3's).
        (1, None, None),
    ],
)
def test_bridge_builds_only_the_named_mtp_layers_detached(
    apply_bridge_runtime_config: Callable,
    runtime_args: argparse.Namespace,
    named: int | None,
    inherited: int | None,
    expected: int | None,
) -> None:
    runtime_args.mtp_num_layers = named
    provider = SimpleNamespace(mtp_num_layers=inherited, mtp_detach_heads=False)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_num_layers == expected
    # Megatron would otherwise train the shared trunk, embedding and output layer on the MTP loss.
    assert provider.mtp_detach_heads is True


def test_a_hybrid_bridge_drops_the_pattern_its_mtp_layers_are_rebuilt_from(
    apply_bridge_runtime_config: Callable, runtime_args: argparse.Namespace
) -> None:
    runtime_args.mtp_num_layers = 0
    provider = SimpleNamespace(mtp_num_layers=1, mtp_hybrid_override_pattern="*-", mtp_detach_heads=False)

    apply_bridge_runtime_config(provider, runtime_args)

    assert (provider.mtp_num_layers, provider.mtp_hybrid_override_pattern) == (None, None)


def test_a_bridge_refuses_a_different_number_of_mtp_layers(
    apply_bridge_runtime_config: Callable, runtime_args: argparse.Namespace
) -> None:
    runtime_args.mtp_num_layers = 2
    provider = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=False)

    with pytest.raises(AssertionError, match="--mtp-num-layers 2, but the model has 1 MTP layers"):
        apply_bridge_runtime_config(provider, runtime_args)
