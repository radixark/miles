"""An HF export carries the source checkpoint's config.json, which declares the speculative draft.

A trainer that trains the draft exports its updated weights; one that does not builds no MTP layers, and
the export takes the draft from the source checkpoint.
"""

import contextlib
import json
import multiprocessing
import os
from argparse import Namespace
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from miles.backends.megatron_utils import hf_export
from miles.backends.training_utils.checkpoint import io as checkpoint_io
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

TRUNK = "model.layers.0.mlp.weight"
DRAFT = "mtp.fc.weight"


def _source(path, tensors, num_hidden_layers=1):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"num_hidden_layers": num_hidden_layers}))
    for i, (name, tensor) in enumerate(tensors.items()):
        safetensors.torch.save_file({name: tensor}, path / f"model-{i}.safetensors")
    weight_map = {name: f"model-{i}.safetensors" for i, name in enumerate(tensors)}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return path


def _exported(export_dir):
    if not (export_dir / "model.safetensors.index.json").exists():
        return safetensors.torch.load_file(export_dir / "model.safetensors")
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    return {name: safetensors.torch.load_file(export_dir / shard)[name] for name, shard in weight_map.items()}


def _export_dir(tmp_path, tensors):
    export = tmp_path / "export"
    export.mkdir()
    # The Bridge exporter writes a single file without an index.
    safetensors.torch.save_file(tensors, export / "model.safetensors")
    return export


@pytest.mark.parametrize(
    "draft",
    [
        "mtp.layers.0.mlp.weight",  # Qwen3.5/3.6, Qwen3-Next
        "model.mtp_layers.0.mlp.weight",  # MiMo
        "model.layers.1.mlp.weight",  # DeepSeek-V3/V4 and GLM NextN layer, past num_hidden_layers=1
    ],
)
def test_a_frozen_draft_comes_from_the_source_checkpoint(tmp_path, draft):
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), draft: torch.ones(2)})
    export = _export_dir(tmp_path, {TRUNK: torch.full((2,), 7.0)})

    hf_export._complete_draft(export, str(source), trained=False)

    exported = _exported(export)
    assert set(exported) == {TRUNK, draft}
    assert torch.equal(exported[draft], torch.ones(2))
    assert torch.equal(exported[TRUNK], torch.full((2,), 7.0))


def test_a_trained_draft_comes_from_training(tmp_path):
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), DRAFT: torch.ones(2)})
    export = _export_dir(tmp_path, {TRUNK: torch.full((2,), 7.0), DRAFT: torch.full((2,), 7.0)})

    hf_export._complete_draft(export, str(source), trained=True)

    assert torch.equal(_exported(export)[DRAFT], torch.full((2,), 7.0))


def test_a_trained_draft_missing_from_the_export_is_an_error(tmp_path):
    """Filling it in from the source would ship the draft the run never trained."""
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), DRAFT: torch.ones(2)})
    export = _export_dir(tmp_path, {TRUNK: torch.full((2,), 7.0)})

    with pytest.raises(AssertionError, match="the trainer trains the speculative draft, but the export lacks"):
        hf_export._complete_draft(export, str(source), trained=True)


def test_other_weights_the_export_lacks_are_not_taken_from_the_source(tmp_path):
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), "lm_head.weight": torch.zeros(2)})
    export = _export_dir(tmp_path, {TRUNK: torch.ones(2)})

    hf_export._complete_draft(export, str(source), trained=False)

    assert set(_exported(export)) == {TRUNK}


@pytest.mark.parametrize("mode", ["raw", "bridge", "lora"])
@pytest.mark.parametrize("trained", [False, True])
def test_every_exporter_meets_the_draft_contract_before_the_export_is_complete(tmp_path, monkeypatch, mode, trained):
    """save_hf_model exports raw models with the publisher, and Bridge and LoRA-merged models with the Bridge."""
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), DRAFT: torch.ones(2)})
    calls = []

    def write_trunk(checkpoint_dir):
        calls.append("weights")
        safetensors.torch.save_file({TRUNK: torch.full((2,), 7.0)}, checkpoint_dir / "model.safetensors")

    def write_adapter(_adapter, _path):
        calls.append("adapter")  # a collective: every rank must reach it before any rank may fail

    monkeypatch.setattr(
        hf_export,
        "get_parallel_state",
        lambda: SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=0)),
    )
    monkeypatch.setattr(hf_export, "get_gloo_group", lambda: None)
    monkeypatch.setattr(hf_export, "is_lora_model", lambda _model: mode == "lora")
    monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(hf_export, "patch_megatron_model", lambda _model: contextlib.nullcontext())
    bridge = SimpleNamespace(save_hf_pretrained=lambda _model, path: write_trunk(path))
    monkeypatch.setattr(hf_export, "_get_hf_bridge", lambda _checkpoint: bridge)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *_args, **_kwargs: None)
    publisher = SimpleNamespace(
        write_model=lambda checkpoint_dir, **_: write_trunk(checkpoint_dir), write_adapter=write_adapter
    )
    args = Namespace(megatron_to_hf_mode="bridge" if mode == "lora" else mode, hf_checkpoint=str(source), save_hf=None)
    args.mtp_num_layers = 1 if trained else 0
    export = tmp_path / "export"

    def save():
        hf_export.save_hf_model(args, 0, [object()], publisher=publisher, path=export, raise_on_error=True)

    if trained:
        # The exporter wrote no draft for a trainer that trained one.
        with pytest.raises(AssertionError, match="the export lacks"):
            save()
        assert not export.exists()
    else:
        save()
        assert (export / HF_EXPORT_COMPLETE_MARKER).exists()
        assert set(_exported(export)) == {TRUNK, DRAFT}
    assert calls == (["weights", "adapter"] if mode == "lora" else ["weights"])


def test_a_failed_draft_check_fails_every_rank_and_marks_nothing(tmp_path, monkeypatch):
    """Rank 0 checks the draft after the last collective, so the other ranks learn of the failure instead of hanging."""
    source = _source(tmp_path / "source", {TRUNK: torch.zeros(2), DRAFT: torch.ones(2)})
    export = tmp_path / "export"

    def save_hf_pretrained(_model, path):
        if torch.distributed.get_rank() == 0:
            safetensors.torch.save_file({TRUNK: torch.full((2,), 7.0)}, path / "model.safetensors")

    monkeypatch.setattr(
        hf_export,
        "get_parallel_state",
        lambda: SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=0)),
    )
    monkeypatch.setattr(hf_export, "get_gloo_group", lambda: None)
    monkeypatch.setattr(checkpoint_io, "get_gloo_group", lambda: None)
    monkeypatch.setattr(hf_export, "is_lora_model", lambda _model: True)
    monkeypatch.setattr(hf_export, "patch_megatron_model", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        hf_export, "_get_hf_bridge", lambda _checkpoint: SimpleNamespace(save_hf_pretrained=save_hf_pretrained)
    )
    # Exporting the adapter is a collective that every rank must reach.
    publisher = SimpleNamespace(write_adapter=lambda _adapter, _path: torch.distributed.barrier())
    args = Namespace(megatron_to_hf_mode="bridge", hf_checkpoint=str(source), save_hf=None, mtp_num_layers=1)

    def rank(rank_id):
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{tmp_path}/store", rank=rank_id, world_size=2
        )
        try:
            hf_export.save_hf_model(args, 0, [object()], publisher=publisher, path=export, raise_on_error=True)
        except (AssertionError, RuntimeError) as error:
            os._exit(0 if "the export lacks" in str(error) else 2)
        os._exit(1)

    children = [multiprocessing.get_context("fork").Process(target=rank, args=(i,)) for i in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=60)
    assert [child.exitcode for child in children] == [0, 0]
    assert not export.exists()
