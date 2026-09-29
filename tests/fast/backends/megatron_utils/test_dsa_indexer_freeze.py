import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def freeze_indexer():
    # Exercise the actual factory helper without importing Megatron's GPU stack.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_freeze_native_dsa_indexer"
    )
    namespace = {"argparse": argparse, "GPTModel": torch.nn.Module}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize(
    ("implementation", "loss_coeff", "explicit_freeze", "expected_frozen"),
    [
        ("megatron", None, False, True),
        ("megatron", 0.0, False, True),
        ("megatron", 0.0, True, True),
        ("megatron", 0.001, False, False),
        ("miles", 0.0, False, False),
    ],
    ids=["default-no-aux", "zero-aux", "explicit-freeze", "train-indexer", "legacy-unchanged"],
)
def test_native_indexer_optimizer_and_ddp_membership(
    freeze_indexer, implementation, loss_coeff, explicit_freeze, expected_frozen
):
    model = torch.nn.Module()
    model.config = SimpleNamespace(dsa_indexer_loss_coeff=loss_coeff)
    model.decoder = torch.nn.Module()
    model.decoder.self_attention = torch.nn.Module()
    model.decoder.self_attention.core_attention = torch.nn.Module()
    indexer = torch.nn.Linear(4, 4)
    model.decoder.self_attention.core_attention.indexer = indexer
    attention_projection = torch.nn.Linear(4, 4)
    model.decoder.self_attention.linear_proj = attention_projection
    args = argparse.Namespace(dsa_impl=implementation, freeze_indexer=explicit_freeze)

    freeze_indexer(args, model)

    # DDP and optimizer construction both enumerate requires_grad parameters.
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert all((id(parameter) in trainable) != expected_frozen for parameter in indexer.parameters())
    assert all(id(parameter) in trainable for parameter in attention_projection.parameters())
