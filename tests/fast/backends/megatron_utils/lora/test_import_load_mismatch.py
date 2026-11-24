"""A client's adapter whose tensors cannot load is refused before the slot resets, as a user error."""

import json

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import find_free_port, init_gloo

from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils import distributed_utils
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

Q, K = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
RUN = AdapterParametrization("base", 2, 4.0, "lora", (Q, K), False)
RUN_SHAPES = {f"{Q}.lora_A": (2, 8), f"{Q}.lora_B": (8, 2), f"{K}.lora_A": (2, 8), f"{K}.lora_B": (4, 2)}


@pytest.fixture(autouse=True)
def gloo(monkeypatch):
    init_gloo(0, 1, port=find_free_port())
    # the cached group belongs to the previous test's destroyed process group
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    init_gloo_group()
    yield
    dist.destroy_process_group()


def _write(path, rank_in_tensors=2):
    shapes = {k: tuple(rank_in_tensors if n == 2 else n for n in s) for k, s in RUN_SHAPES.items()}
    write_peft_adapter(path, {k: torch.randn(s) for k, s in shapes.items()}, RUN, TARGETS, lora_dropout=0.0)
    # the config keeps the run's rank whatever the tensors hold
    config = json.loads((path / "adapter_config.json").read_text())
    (path / "adapter_config.json").write_text(json.dumps(config | {"r": 2}))


def _mismatch(path):
    return checkpoint.check_load_source(None, 0, str(path), RUN, TARGETS, False, run_shapes=RUN_SHAPES)[0]


def test_a_fitting_adapter_loads(tmp_path):
    _write(tmp_path)
    assert _mismatch(tmp_path) is None


def test_a_truncated_tensor_file_is_refused(tmp_path):
    _write(tmp_path)
    tensors = tmp_path / "adapter_model.safetensors"
    tensors.write_bytes(tensors.read_bytes()[:40])
    assert "cannot load into this model" in _mismatch(tmp_path)


def test_tensors_of_another_rank_than_the_config_are_refused(tmp_path):
    _write(tmp_path, rank_in_tensors=4)
    assert "tensor shapes differ" in _mismatch(tmp_path)
