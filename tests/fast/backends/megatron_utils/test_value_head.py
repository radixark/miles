import sys
import types

import pytest
import torch


@pytest.fixture
def value_head(monkeypatch):
    core = types.ModuleType("megatron.core")
    core.tensor_parallel = types.SimpleNamespace(gather_from_sequence_parallel_region=lambda x: x)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.delitem(sys.modules, "miles.backends.megatron_utils.value_head", raising=False)
    import miles.backends.megatron_utils.value_head as module

    return module


def _chunk(*, post_process: bool, hidden_size: int = 8) -> torch.nn.Module:
    chunk = torch.nn.Module()
    chunk.post_process = post_process
    chunk.config = types.SimpleNamespace(hidden_size=hidden_size, sequence_parallel=False)
    chunk.output_layer = torch.nn.Linear(hidden_size, 32)
    return chunk


def test_head_replaces_output_layer_only_on_last_stage_chunks(value_head):
    chunks = [_chunk(post_process=False), _chunk(post_process=True)]
    value_head.attach_value_head(chunks)

    assert isinstance(chunks[0].output_layer, torch.nn.Linear)
    assert not isinstance(chunks[0].output_layer, value_head.LinearForLastLayer)
    head = chunks[1].output_layer
    assert isinstance(head, value_head.LinearForLastLayer)
    assert head.weight.shape == (1, 8)
    assert head.weight.requires_grad and head.bias.requires_grad


def test_head_takes_the_chunk_config(value_head):
    chunk = _chunk(post_process=True, hidden_size=16)
    value_head.attach_value_head(chunk)

    assert chunk.output_layer.sequence_parallel is False
    logits, bias = chunk.output_layer(torch.zeros(3, 2, 16))
    assert logits.shape == (3, 2, 1) and bias is None
