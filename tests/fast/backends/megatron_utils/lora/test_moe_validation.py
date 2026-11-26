"""Exercise the resolved-provider validation without GPU-only Megatron imports."""

import ast
import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.lora.utils import targets_expert_leaves


@pytest.fixture(scope="module")
def validate_moe():
    path = Path(__file__).resolve().parents[5] / "miles/backends/megatron_utils/lora/bridge.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_multi_lora_moe_support"
    )
    namespace = {
        "Namespace": Namespace,
        "logger": logging.getLogger(__name__),
        "targets_expert_leaves": targets_expert_leaves,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def provider(*, gated):
    return SimpleNamespace(num_moe_experts=8, gated_linear_unit=gated, moe_grouped_gemm=True)


@pytest.mark.parametrize("prefix", ["", "model.layers.0.mlp.experts.0."])
@pytest.mark.parametrize(
    "gated, targets",
    [
        (False, ["up_proj", "down_proj"]),
        (False, ["gate_up_proj", "down_proj"]),
        (True, ["gate_proj", "up_proj", "down_proj"]),
        (True, ["gate_up_proj", "down_proj"]),
    ],
)
def test_complete_expert_targets(validate_moe, prefix, gated, targets):
    args = Namespace(hf_lora_targets=[prefix + target for target in targets])
    validate_moe(args, provider(gated=gated))


@pytest.mark.parametrize(
    "gated, targets",
    [
        (False, ["up_proj"]),
        (False, ["down_proj"]),
        (True, ["up_proj", "down_proj"]),
        (True, ["gate_proj", "down_proj"]),
        (True, ["gate_proj", "up_proj"]),
    ],
)
def test_incomplete_expert_targets_are_rejected(validate_moe, gated, targets):
    with pytest.raises(AssertionError, match="requires all of"):
        validate_moe(Namespace(hf_lora_targets=targets), provider(gated=gated))


@pytest.mark.parametrize("gated", [False, True])
def test_attention_only_adapters_do_not_require_expert_targets(validate_moe, gated):
    validate_moe(Namespace(hf_lora_targets=["q_proj", "v_proj"]), provider(gated=gated))


@pytest.mark.parametrize(
    "setting, value, message",
    [
        ("expert_tensor_parallel_size", 2, "expert_tensor_parallel_size=1"),
        ("moe_grouped_gemm", False, "moe_grouped_gemm=True"),
        ("fp8", "hybrid", "does not support fp8/fp4"),
        ("moe_pad_expert_input_to_capacity", True, "moe-pad-expert-input-to-capacity"),
        ("moe_permute_fusion", True, "moe_permute_fusion=False"),
    ],
)
def test_ungated_experts_still_enforce_backend_constraints(validate_moe, setting, value, message):
    config = provider(gated=False)
    setattr(config, setting, value)
    with pytest.raises(AssertionError, match=message):
        validate_moe(Namespace(hf_lora_targets=["up_proj", "down_proj"]), config)
