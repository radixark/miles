import errno
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


def _legacy_source(monkeypatch, saved_shape):
    monkeypatch.setattr(
        checkpoint, "check_adapter_source", lambda *_, **__: AdapterSourceKind.LEGACY_TINKER_CHECKPOINT
    )
    monkeypatch.setattr(checkpoint, "_read_legacy_meta", lambda *_: {})
    metadata = SimpleNamespace(state_dict_metadata={"layer.adapter.w": SimpleNamespace(size=torch.Size(saved_shape))})
    monkeypatch.setattr(checkpoint.dcp, "FileSystemReader", lambda _: SimpleNamespace(read_metadata=lambda: metadata))
    monkeypatch.setattr(
        checkpoint,
        "_slot_weights_sharded_state_dict",
        lambda *_: {"w": SimpleNamespace(key="layer.adapters.0.w", global_shape=(32, 6))},
    )


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_a_legacy_checkpoint_with_the_slots_keys_loads(gloo, monkeypatch, load_optimizer):
    _legacy_source(monkeypatch, saved_shape=(32, 6))
    assert checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer) is None


def test_a_legacy_checkpoint_padded_to_another_max_rank_is_a_mismatch(gloo, monkeypatch):
    _legacy_source(monkeypatch, saved_shape=(64, 6))
    mismatch = checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer=False)
    assert mismatch is not None and "layer.adapter.w: saved (64, 6), slot (32, 6)" in mismatch


def test_a_model_side_failure_is_a_server_error_not_a_mismatch(gloo, monkeypatch):
    _legacy_source(monkeypatch, saved_shape=(32, 6))

    def broken(*_):
        raise KeyError("slot 0 has no adapters")

    monkeypatch.setattr(checkpoint, "_slot_weights_sharded_state_dict", broken)
    # a mismatch is the user's 400; a bug in reading this gateway's own model must fail loudly instead
    with pytest.raises(RuntimeError, match="slot 0 has no adapters"):
        checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer=False)


@pytest.mark.parametrize(
    "error", [OSError(errno.EIO, "Input/output error"), TypeError("library bug")], ids=["storage_fault", "bug"]
)
def test_a_failure_that_is_not_the_sources_fault_is_a_server_error(gloo, monkeypatch, error):
    def failing(*_, **__):
        raise error

    monkeypatch.setattr(checkpoint, "check_adapter_source", failing)
    with pytest.raises(RuntimeError, match=str(error.args[-1])):
        checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer=False)


def test_a_missing_source_is_the_users_mismatch(gloo, tmp_path):
    mismatch = checkpoint.find_load_mismatch(
        None, 0, str(tmp_path / "absent"), UNSHARED, TARGETS, load_optimizer=False
    )
    assert mismatch is not None and "cannot load into this model" in mismatch


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


def test_a_legacy_checkpoint_is_read_before_anything_is_loaded_into_the_slot(tmp_path, monkeypatch):
    run = AdapterParametrization("base", 4, 8.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
    (tmp_path / "META.json").write_text('{"base_model": "base", "lora_rank": 4, "lora_alpha": 16.0}')
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_legacy_slot", lambda *args: loaded.append(args))
    monkeypatch.setattr(checkpoint, "_slot_weights_sharded_state_dict", lambda model, slot: {"w": (model, slot)})
    loader = checkpoint.LegacyTinkerCheckpointLoader("model", SimpleNamespace(slot=2), run, load_optimizer=True)

    assert loader.parametrization(tmp_path) == replace(run, lora_alpha=16.0)
    prepared = loader.prepare(tmp_path)
    assert loaded == [], "reading the source and its shells loads nothing"
    loader.apply(prepared)
    assert loaded == [({"w": ("model", 2)}, loader._optimizer, str(tmp_path), True)]
