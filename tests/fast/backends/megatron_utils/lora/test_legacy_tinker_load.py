import errno
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import find_free_port, init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import CheckedSource, HeadRows
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
from miles.utils import distributed_utils
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

Q, K = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
UNSHARED = AdapterParametrization("m", 2, 4.0, "canonical_lora", (Q, K), False)
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})


def _legacy_metadata(saved_shape, *, with_optimizer=True):
    keys = {"layer.adapter.w": SimpleNamespace(size=torch.Size(saved_shape))}
    if with_optimizer:
        keys["optimizer.state.exp_avg.layer.adapter.w"] = SimpleNamespace(size=torch.Size(saved_shape))
    return SimpleNamespace(state_dict_metadata=keys)


def _legacy_source(monkeypatch, saved_shape, *, with_optimizer=True):
    monkeypatch.setattr(
        checkpoint, "check_adapter_source", lambda *_, **__: CheckedSource(AdapterSourceKind.LEGACY_TINKER_CHECKPOINT)
    )
    monkeypatch.setattr(checkpoint, "_read_legacy_meta", lambda *_: {})
    metadata = _legacy_metadata(saved_shape, with_optimizer=with_optimizer)
    monkeypatch.setattr(checkpoint.dcp, "FileSystemReader", lambda _: SimpleNamespace(read_metadata=lambda: metadata))
    monkeypatch.setattr(
        checkpoint,
        "_slot_weights_sharded_state_dict",
        lambda *_: {"w": SimpleNamespace(key="layer.adapters.0.w", global_shape=(32, 6))},
    )


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_a_legacy_checkpoint_with_the_slots_keys_loads(gloo, monkeypatch, load_optimizer):
    _legacy_source(monkeypatch, saved_shape=(32, 6))
    assert checkpoint.check_load_source(None, 0, "/ckpt", None, None, load_optimizer, run_shapes={})[0] is None


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_a_legacy_checkpoint_without_optimizer_state_is_a_mismatch(gloo, monkeypatch, load_optimizer):
    _legacy_source(monkeypatch, saved_shape=(32, 6), with_optimizer=False)
    mismatch = checkpoint.check_load_source(None, 0, "/ckpt", None, None, load_optimizer, run_shapes={})[0]
    assert mismatch is not None and "holds no optimizer state, which every legacy slot load reads" in mismatch


def test_a_legacy_checkpoint_padded_to_another_max_rank_is_a_mismatch(gloo, monkeypatch):
    _legacy_source(monkeypatch, saved_shape=(64, 6))
    mismatch = checkpoint.check_load_source(None, 0, "/ckpt", None, None, load_optimizer=False, run_shapes={})[0]
    assert mismatch is not None and "layer.adapter.w: saved (64, 6), slot (32, 6)" in mismatch


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EIO, "Input/output error"),
        AssertionError("backend invariant"),
        ValueError("backend value bug"),
    ],
    ids=["storage_fault", "assertion", "value_error"],
)
def test_a_failure_that_is_not_the_sources_fault_is_a_server_error(gloo, monkeypatch, error):
    def failing(*_, **__):
        raise error

    monkeypatch.setattr(checkpoint, "check_adapter_source", failing)
    with pytest.raises(RuntimeError, match=str(error.args[-1])):
        checkpoint.check_load_source(None, 0, "/ckpt", None, None, load_optimizer=False, run_shapes={})[0]


def _check_missing_source(rank, world_size, port, *, path):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    checked_on = []
    check = checkpoint.check_adapter_source

    def counting_check(*args, **kwargs):
        checked_on.append(rank)
        return check(*args, **kwargs)

    checkpoint.check_adapter_source = counting_check
    mismatch = checkpoint.check_load_source(None, 0, path, UNSHARED, TARGETS, load_optimizer=False, run_shapes={})[0]
    assert checked_on == ([0] if rank == 0 else []), "only rank 0 reads the source"
    assert mismatch is not None and "cannot load into this model" in mismatch
    dist.destroy_process_group()


def test_a_missing_source_is_the_users_mismatch_on_every_rank(tmp_path):
    """Rank 0 reads and classifies the source; every rank returns its verdict."""
    run_multiprocess(partial(_check_missing_source, path=str(tmp_path / "absent")))


@pytest.fixture
def gloo(monkeypatch):
    init_gloo(0, 1, port=find_free_port())
    # the cached group belongs to the previous test's destroyed process group
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    init_gloo_group()
    yield
    dist.destroy_process_group()


def test_a_source_the_weight_conversion_refuses_is_a_mismatch(gloo, tmp_path):
    weights = {f"{Q}.lora_A": torch.zeros(2, 6), f"{K}.lora_A": torch.ones(2, 6)}
    weights |= {f"{m}.lora_B": torch.zeros(4, 2) for m in (Q, K)}
    write_peft_adapter(tmp_path, weights, UNSHARED, TARGETS, lora_dropout=0.0)
    shared = replace(UNSHARED, shared_lora_a_groups=((Q, K),))
    mismatch = checkpoint.check_load_source(
        None, 0, str(tmp_path), shared, TARGETS, load_optimizer=False, run_shapes={}
    )[0]
    assert mismatch is not None and "cannot share A" in mismatch


def test_a_head_over_the_served_vocab_is_no_mismatch(gloo, tmp_path):
    """Official Tinker and HF PEFT exports carry the HF vocab's head rows; the load keeps the served ones."""
    head = replace(UNSHARED, target_modules=("lm_head",))
    write_peft_adapter(
        tmp_path,
        {"lm_head.lora_A": torch.zeros(2, 6), "lm_head.lora_B": torch.zeros(8, 2)},
        head,
        TARGETS,
        lora_dropout=0.0,
    )
    run_shapes = {"lm_head.lora_A": (2, 6), "lm_head.lora_B": (6, 2)}
    assert (
        checkpoint.check_load_source(
            None,
            0,
            str(tmp_path),
            head,
            TARGETS,
            load_optimizer=False,
            run_shapes=run_shapes,
            head_rows=HeadRows("lm_head.lora_B", max_source_rows=8),
        )[0]
        is None
    )
    too_many = checkpoint.check_load_source(
        None,
        0,
        str(tmp_path),
        head,
        TARGETS,
        load_optimizer=False,
        run_shapes=run_shapes,
        head_rows=HeadRows("lm_head.lora_B", max_source_rows=7),
    )[0]
    assert too_many is not None and "the source has 8 rows" in too_many


def test_an_unreadable_source_is_a_mismatch(gloo, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"not safetensors")
    mismatch = checkpoint.check_load_source(
        None, 0, str(tmp_path), UNSHARED, TARGETS, load_optimizer=False, run_shapes={}
    )[0]
    assert mismatch is not None and "cannot load into this model" in mismatch


def test_a_legacy_layout_mismatch_names_the_gateway_flag(tmp_path):
    # META.json from before the layout was recorded: fused, per-expert
    (tmp_path / "META.json").write_text('{"base_model": "m", "lora_rank": 2, "lora_alpha": 4.0}')
    with pytest.raises(AdapterSourceError, match="launched with --lora-type lora"):
        checkpoint._read_legacy_meta(tmp_path, UNSHARED)


def test_a_legacy_meta_without_a_required_field_is_the_users_mismatch(gloo, tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpoint, "check_adapter_source", lambda *_, **__: CheckedSource(AdapterSourceKind.LEGACY_TINKER_CHECKPOINT)
    )
    (tmp_path / "META.json").write_text('{"base_model": "m", "lora_alpha": 4.0}')
    mismatch = checkpoint.check_load_source(
        None, 0, str(tmp_path), UNSHARED, TARGETS, load_optimizer=False, run_shapes={}
    )[0]
    assert mismatch is not None and "lacks ['lora_rank']" in mismatch


def test_a_legacy_checkpoint_is_read_before_anything_is_loaded_into_the_slot(tmp_path, monkeypatch):
    run = AdapterParametrization("base", 4, 8.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
    (tmp_path / "META.json").write_text('{"base_model": "base", "lora_rank": 4, "lora_alpha": 16.0}')
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_legacy_slot", lambda *args: loaded.append(args))
    monkeypatch.setattr(checkpoint, "_slot_weights_sharded_state_dict", lambda model, slot: {"w": (model, slot)})
    optimizer = SimpleNamespace(slot=2, sharded_state=lambda weights, is_loading: {"optim": is_loading})
    loader = checkpoint.LegacyTinkerCheckpointLoader("model", optimizer, run, load_optimizer=True)

    saved = {"metadata": _legacy_metadata((4, 8))}
    monkeypatch.setattr(
        checkpoint.dcp, "FileSystemReader", lambda _: SimpleNamespace(read_metadata=lambda: saved["metadata"])
    )
    assert loader.parametrization(tmp_path) == (replace(run, lora_alpha=16.0), True)
    saved["metadata"] = _legacy_metadata((4, 8), with_optimizer=False)
    with pytest.raises(AdapterSourceError, match="holds no optimizer state"):
        loader.parametrization(tmp_path)
    prepared = loader.prepare(tmp_path)
    assert loaded == [], "reading the source and its shells loads nothing"
    loader.apply(prepared)
    shells = {"adapter_weights": {"w": ("model", 2)}, "adapter_optimizer": {"optim": True}}
    assert loaded == [(shells, optimizer, str(tmp_path), True)]
