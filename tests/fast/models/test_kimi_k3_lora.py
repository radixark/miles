from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from miles_plugins.models.kimi_k3.lora import (
    KimiK3LoRAAdapter,
    _enable_full_recompute_input_grads,
    _grouped_linear,
    export_kimi_k3_lora_hf_chunks,
)


def _parameter(*shape):
    return nn.Parameter(torch.arange(torch.tensor(shape).prod()).reshape(shape).float())


def _mla_attention_adapter():
    adapter = KimiK3LoRAAdapter("mla_attention", "language_model.model.layers.3.self_attn.")
    adapter.register_parameter("q_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("q_a_lora_B", _parameter(4, 2))
    adapter.register_parameter("kv_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("kv_a_lora_B", _parameter(6, 2))
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _expert_adapter():
    adapter = KimiK3LoRAAdapter(
        "experts",
        "language_model.model.layers.4.block_sparse_moe.experts.",
    )
    adapter.register_parameter("w1_lora_A", _parameter(2, 8))
    adapter.register_parameter("w3_lora_A", _parameter(2, 8))
    adapter.register_parameter("w1_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w3_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w2_lora_A", _parameter(3, 2, 5))
    adapter.register_parameter("w2_lora_B", _parameter(8, 2))
    return adapter


def _shared_expert_adapter():
    adapter = KimiK3LoRAAdapter(
        "shared_experts",
        "language_model.model.layers.4.block_sparse_moe.shared_experts.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _dense_adapter():
    adapter = KimiK3LoRAAdapter(
        "dense_mlp",
        "language_model.model.layers.3.mlp.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _kda_attention_adapter():
    adapter = KimiK3LoRAAdapter(
        "kda_attention",
        "language_model.model.layers.4.self_attn.",
    )
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _model_with_adapters(*, include_shared_experts=True):
    mla_attention = _mla_attention_adapter()
    dense = _dense_adapter()
    kda_attention = _kda_attention_adapter()
    experts = _expert_adapter()
    shared_experts = _shared_expert_adapter()
    adapters = [mla_attention, dense, kda_attention, experts]
    if include_shared_experts:
        adapters.append(shared_experts)

    model = nn.Module()
    model.adapters = nn.ModuleList(adapters)
    model.decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_number=4,
                self_attention=SimpleNamespace(is_kda=False),
                mlp=SimpleNamespace(),
            ),
            SimpleNamespace(
                layer_number=5,
                self_attention=SimpleNamespace(is_kda=True),
                mlp=SimpleNamespace(
                    experts=SimpleNamespace(),
                    shared_experts=SimpleNamespace(),
                ),
            ),
        ]
    )
    return model


def test_native_export_is_chunked_by_adapter(monkeypatch):
    """A wrong expert dim is a shape mismatch in SGLang's LoRA pool; a wrong HF name is silently dropped."""
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters()
    chunks = list(export_kimi_k3_lora_hf_chunks([model]))

    assert len(chunks) == 5
    attention = dict(chunks[0])
    assert attention["language_model.model.layers.3.self_attn.q_a_proj.lora_A.weight"].shape == (2, 8)
    assert attention["language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.lora_B.weight"].shape == (
        6,
        2,
    )
    assert attention["language_model.model.layers.3.self_attn.o_proj.lora_A.weight"].shape == (2, 3)

    experts = dict(chunks[3])
    prefix = "language_model.model.layers.4.block_sparse_moe.experts."
    assert experts[f"{prefix}w1.lora_A.weight"].shape == (1, 2, 8)
    assert experts[f"{prefix}w1.lora_B.weight"].shape == (3, 5, 2)
    assert experts[f"{prefix}w2.lora_A.weight"].shape == (3, 2, 5)
    assert experts[f"{prefix}w2.lora_B.weight"].shape == (1, 8, 2)

    shared_experts = dict(chunks[4])
    prefix = "language_model.model.layers.4.block_sparse_moe.shared_experts."
    assert shared_experts[f"{prefix}gate_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}gate_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}up_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}up_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}down_proj.lora_A.weight"].shape == (2, 5)
    assert shared_experts[f"{prefix}down_proj.lora_B.weight"].shape == (8, 2)

    backups = {id(parameter): torch.full_like(parameter, 17) for parameter in model.parameters()}
    for chunk in export_kimi_k3_lora_hf_chunks(
        [model], materialize_parameter=lambda parameter: backups[id(parameter)]
    ):
        for _name, tensor in chunk:
            torch.testing.assert_close(tensor, torch.full_like(tensor, 17))


def test_native_export_rejects_missing_shared_expert_adapter(monkeypatch):
    """A layer without its adapter must fail the export, not ship a partial adapter SGLang accepts."""
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters(include_shared_experts=False)

    with pytest.raises(RuntimeError, match="adapter layout is incomplete"):
        list(export_kimi_k3_lora_hf_chunks([model]))


def test_grouped_linear_uses_expert_token_boundaries():
    """Wrong boundaries apply expert i's adapter to expert j's tokens without any error."""
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    output = _grouped_linear(inputs, weights, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[1.0], [4.0], [6.0]]))


def test_full_recompute_keeps_native_lora_in_autograd_graph():
    """Under full recompute with a frozen base the segment input carries no grad, so every LoRA
    gradient is zero; the embedding hook must fix that in training and stay out of eval."""
    model = nn.Module()
    model.embedding = nn.Embedding.from_pretrained(torch.ones(8, 4), freeze=True)
    model.adapter = nn.Parameter(torch.ones(4, 4))
    model.config = SimpleNamespace(recompute_granularity="full")
    model.pre_process = True
    model.embedding.requires_grad_(False)
    _enable_full_recompute_input_grads(model)

    model.train()
    hidden_states = model.embedding(torch.tensor([[1, 2]]))
    output = checkpoint(lambda inputs: inputs @ model.adapter, hidden_states, use_reentrant=True)
    output.sum().backward()

    assert hidden_states.requires_grad
    assert model.embedding.weight.grad is None
    torch.testing.assert_close(model.adapter.grad, torch.full_like(model.adapter, 2.0))

    model.eval()
    assert not model.embedding(torch.tensor([[1, 2]])).requires_grad
