import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from miles.backends.megatron_utils.named_weights import _maybe_get_cpu_backup, named_params_and_buffers
from miles.backends.training_utils.model_companion import ModelCompanion


@pytest.fixture
def memory_saver(monkeypatch: pytest.MonkeyPatch) -> Mock:
    saver = Mock()
    saver.get_cpu_backup = Mock(side_effect=AssertionError("get_cpu_backup must not see a non-CUDA tensor"))
    module = ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    return saver


def test_a_cpu_tensor_is_returned_without_consulting_the_memory_saver(memory_saver: Mock) -> None:
    """Host-resident parameters have no memory-saver backup, so they pass through untouched."""
    tensor = torch.zeros((), dtype=torch.int64)

    assert _maybe_get_cpu_backup(tensor) is tensor
    memory_saver.get_cpu_backup.assert_not_called()


def test_a_cuda_tensor_is_translated_to_its_host_backup(memory_saver: Mock) -> None:
    """Device-resident parameters are looked up in the memory saver's host backup."""
    backup = torch.zeros((), dtype=torch.int64)
    memory_saver.get_cpu_backup = Mock(return_value=backup)
    device_tensor: Any = SimpleNamespace(is_cuda=True)

    assert _maybe_get_cpu_backup(device_tensor) is backup
    memory_saver.get_cpu_backup.assert_called_once_with(device_tensor, zero_copy=True)


def test_a_cuda_tensor_without_a_backup_is_returned_unchanged(memory_saver: Mock) -> None:
    """A device tensor the memory saver does not manage keeps its live storage."""
    memory_saver.get_cpu_backup = Mock(return_value=None)
    device_tensor: Any = SimpleNamespace(is_cuda=True)

    assert _maybe_get_cpu_backup(device_tensor) is device_tensor


def test_translating_to_host_backups_keeps_the_companion_parameters(memory_saver: Mock) -> None:
    """The model companion lives on the host and survives the GPU-to-CPU translation."""
    chunk = torch.nn.Module()
    chunk.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))
    chunk.model_companion.weight_version.fill_(5)

    named = dict(
        named_params_and_buffers(
            Namespace(),
            [chunk],
            convert_to_global_name=False,
            translate_gpu_to_cpu=True,
            include_model_companion=True,
        )
    )

    version = named["vp_stages.0.model_companion.weight_version"]
    assert version.data_ptr() == chunk.model_companion.weight_version.data_ptr()
    assert version.item() == 5
    memory_saver.get_cpu_backup.assert_not_called()


def test_sequential_mlp_experts_get_grouped_names_with_global_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """EP rank 1 of 2 with 8 experts: local expert 1 is global expert 5, in decoder and MTP layers alike."""
    from miles.backends.megatron_utils import named_weights as nw

    monkeypatch.setattr(nw, "get_parallel_state", lambda: SimpleNamespace(ep=SimpleNamespace(size=2, rank=1)))
    monkeypatch.setattr(nw, "get_transformer_layer_offset", lambda config, *args, **kwargs: 0)

    class Chunk(torch.nn.Module):
        config = None

        def named_parameters(self, *args, **kwargs):
            yield "module.module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight", torch.zeros(1)
            yield "module.module.mtp.layers.0.transformer_layer.mlp.experts.local_experts.1.linear_fc2.weight", torch.zeros(
                1
            )

        def named_buffers(self, *args, **kwargs):
            return iter(())

    names = [
        name for name, _ in nw._named_params_and_buffers_global(Namespace(backend=Namespace(num_experts=8)), [Chunk()])
    ]

    assert names == [
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight5",
        "module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight5",
    ]
