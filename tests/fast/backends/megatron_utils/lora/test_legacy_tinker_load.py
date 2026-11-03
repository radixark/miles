from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import find_free_port, init_gloo

from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

Q, K = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
UNSHARED = AdapterParametrization("m", 2, 4.0, "canonical_lora", (Q, K), False)
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})


@pytest.fixture
def legacy_source(monkeypatch):
    init_gloo(0, 1, port=find_free_port())
    init_gloo_group()
    monkeypatch.setattr(
        checkpoint, "check_adapter_source", lambda *_, **__: AdapterSourceKind.LEGACY_TINKER_CHECKPOINT
    )
    monkeypatch.setattr(checkpoint, "_read_legacy_meta", lambda *_: {})
    metadata = SimpleNamespace(state_dict_metadata={"layer.adapter.w": None})
    monkeypatch.setattr(checkpoint.dcp, "FileSystemReader", lambda _: SimpleNamespace(read_metadata=lambda: metadata))
    monkeypatch.setattr(
        checkpoint, "_slot_weights_sharded_state_dict", lambda *_: {"w": SimpleNamespace(key="layer.adapters.0.w")}
    )
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_a_legacy_checkpoint_with_the_slots_keys_loads(legacy_source, load_optimizer):
    assert checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer) is None


@pytest.fixture
def gloo():
    init_gloo(0, 1, port=find_free_port())
    init_gloo_group()
    yield
    dist.destroy_process_group()


def test_a_source_the_weight_conversion_refuses_is_a_mismatch(gloo, tmp_path):
    weights = {f"{Q}.lora_A": torch.zeros(2, 6), f"{K}.lora_A": torch.ones(2, 6)}
    weights |= {f"{m}.lora_B": torch.zeros(4, 2) for m in (Q, K)}
    write_peft_adapter(tmp_path, weights, UNSHARED, TARGETS)
    shared = replace(UNSHARED, shared_lora_a_groups=((Q, K),))
    mismatch = checkpoint.find_load_mismatch(None, 0, str(tmp_path), shared, TARGETS, load_optimizer=False)
    assert mismatch is not None and "cannot share A" in mismatch


def test_an_unreadable_source_is_a_mismatch(gloo, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"not safetensors")
    mismatch = checkpoint.find_load_mismatch(None, 0, str(tmp_path), UNSHARED, TARGETS, load_optimizer=False)
    assert mismatch is not None and "cannot load into this model" in mismatch


def test_an_in_model_source_without_a_loader_is_a_mismatch(gloo, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    torch.save({}, tmp_path / "adapter_megatron_rank0.pt")
    mismatch = checkpoint.find_load_mismatch(None, 0, str(tmp_path), UNSHARED, TARGETS, load_optimizer=False)
    assert mismatch is not None and "legacy_rank_shard source, which this backend cannot load" in mismatch


def test_a_legacy_layout_mismatch_names_the_gateway_flag(tmp_path):
    # META.json from before the layout was recorded: fused, per-expert
    (tmp_path / "META.json").write_text('{"base_model": "m", "lora_rank": 2, "lora_alpha": 4.0}')
    with pytest.raises(ValueError, match="launched with --lora-type lora"):
        checkpoint._read_legacy_meta(tmp_path, UNSHARED)
