"""Exercise the LoRA bridge builder's output contract without GPU dependencies."""

import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize("returns_tuple", [False, True])
@pytest.mark.parametrize("num_chunks", [1, 2])
def test_lora_bridge_forward_preserves_logits_and_gradients(monkeypatch, returns_tuple, num_chunks):
    # Exercise the production builder, including its final wrapping step. Extract
    # it as in test_bridge_mtp_detachment.py to avoid GPU-only Megatron imports.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/lora/bridge.py"
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_setup_lora_model_via_bridge", "_ensure_model_list"}
    ]

    class BridgeChunk(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.last_logits = None

        def forward(self, inputs, *, loss_mask):
            self.last_logits = inputs * self.scale
            return (self.last_logits, loss_mask) if returns_tuple else self.last_logits

    chunks = [BridgeChunk(index + 1) for index in range(num_chunks)]
    provider = MagicMock()
    provider.provide_distributed_model.return_value = chunks
    auto_bridge = MagicMock()
    auto_bridge.from_hf_pretrained.return_value.to_megatron_provider.return_value = provider
    lora = MagicMock()
    lora.side_effect = lambda model_chunks, training: model_chunks

    for name, attrs in (
        ("megatron.bridge", {"AutoBridge": auto_bridge}),
        ("megatron.bridge.training.config", {"DistributedDataParallelConfig": MagicMock()}),
        ("miles.backends.megatron_utils.lora.utils", {"create_lora_instance": lambda args: lora}),
    ):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    namespace = {
        "__name__": "miles.backends.megatron_utils.lora.bridge",
        "__package__": "miles.backends.megatron_utils.lora",
        "Namespace": Namespace,
        "load_hf_config": lambda path: SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
        "is_multi_lora_enabled": lambda args: False,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    args = Namespace(
        hf_checkpoint="unused",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        gradient_accumulation_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=[],
        distribute_saved_activations=False,
        attention_backend="auto",
        optimizer="adam",
        accumulate_allreduce_grads_in_fp32=True,
        offload_train=False,
    )
    models = namespace["_setup_lora_model_via_bridge"](args)
    assert models is chunks
    for index, model in enumerate(models):
        inputs = torch.ones(2, 3, requires_grad=True)
        output = model(inputs, loss_mask=torch.ones(2))
        # The loss reads dtype on this output; a tuple fails before CE runs.
        assert output.dtype == inputs.dtype
        assert output is model.last_logits
        torch.testing.assert_close(output, inputs * (index + 1))
        output.sum().backward()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, index + 1))
