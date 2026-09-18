import json
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
import safetensors.torch
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cpu_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from miles.backends.torchtitan_utils import hf_export

register_cpu_ci(est_time=75, suite="stage-a-cpu")


class _Adapter:
    def __init__(self, *, tied=False):
        self.tied = tied
        self.fqn_to_index_mapping = {"model.embed_tokens.weight": 1, "lm_head.weight": 2}

    def to_hf(self, state_dict):
        weights = dict(state_dict)
        if self.tied:
            weights.pop("lm_head.weight", None)
            if self.fqn_to_index_mapping:
                self.fqn_to_index_mapping.pop("lm_head.weight", None)
        return weights


class _Checkpointer:
    def __init__(self, state_dict, adapter):
        self.states = {"model": SimpleNamespace(state_dict=lambda: state_dict)}
        self.sd_adapter = adapter

    def dcp_save(self, state_dict, checkpoint_id, async_mode, to_hf):
        assert async_mode == "disabled" and to_hf
        # CPU CI has no Titan; exercise its storage backend with real distributed tensors.
        dcp.save(
            self.sd_adapter.to_hf(state_dict),
            storage_writer=dcp.HuggingFaceStorageWriter(
                path=checkpoint_id, save_distributed=True, enable_consolidation=True
            ),
        )


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    rendezvous = tmp_path_factory.mktemp("hf-export") / "rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _source(path, *, tied=False, towers=False):
    path.mkdir()
    config = {"model_type": "qwen3", "tie_word_embeddings": tied}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text('{"tokenizer": "fixture"}')
    (path / ".complete").touch()
    weights = {
        "model.embed_tokens.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "lm_head.weight": torch.ones(4, 4),
    }
    if tied:
        del weights["lm_head.weight"]
    if towers:
        weights["model.visual.proj.weight"] = torch.full((2, 2), 3.0)
    safetensors.torch.save_file(weights, path / "model-00001-of-00001.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model-00001-of-00001.safetensors")})
    )
    return weights


def _load_export(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for shard in set(index["weight_map"].values()):
        weights.update(safetensors.torch.load_file(path / shard))
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in weights.values()
    )
    assert set(weights) == set(index["weight_map"])
    assert (path / ".complete").exists()
    return weights


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_exports_current_weights_and_assets(tmp_path, process_group, tied):
    source, destination = tmp_path / "source", tmp_path / "export"
    original = _source(source, tied=tied)
    trained = {name: tensor + 1 for name, tensor in original.items()}
    if tied:
        trained["lm_head.weight"] = trained["model.embed_tokens.weight"]
    adapter = _Adapter(tied=tied)
    hf_export.export_hf(_Checkpointer(trained, adapter), hf_checkpoint=str(source), path=str(destination))

    exported = _load_export(destination)
    assert exported.keys() == original.keys()
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, trained[name])
    assert (destination / "config.json").read_text() == (source / "config.json").read_text()
    assert (destination / "tokenizer.json").read_text() == (source / "tokenizer.json").read_text()
    assert not (destination / "sharded").exists()


@pytest.mark.parametrize("indexed", [False, True], ids=["single-file", "indexed"])
def test_multimodal_checkpoint_is_rejected(tmp_path, process_group, indexed):
    source, destination = tmp_path / "source", tmp_path / "export"
    weights = _source(source, towers=True)
    if not indexed:
        (source / "model.safetensors.index.json").unlink()
    with pytest.raises(NotImplementedError, match="multimodal"):
        hf_export.export_hf(_Checkpointer(weights, _Adapter()), hf_checkpoint=str(source), path=str(destination))
    assert not destination.exists()


@pytest.mark.parametrize("reexport", [False, True], ids=["new", "existing"])
def test_failed_metadata_copy_never_publishes_a_marker(tmp_path, process_group, monkeypatch, reexport):
    source, destination = tmp_path / "source", tmp_path / "export"
    weights = _source(source)
    checkpointer = _Checkpointer(weights, _Adapter())
    if reexport:
        hf_export.export_hf(checkpointer, hf_checkpoint=str(source), path=str(destination))

    def fail_copy(*args):
        raise OSError("disk full")

    monkeypatch.setattr(hf_export.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        hf_export.export_hf(checkpointer, hf_checkpoint=str(source), path=str(destination))
    assert (destination / "model.safetensors.index.json").exists()
    assert not (destination / ".complete").exists()


def test_reexport_replaces_stale_shards(tmp_path, process_group):
    source, destination = tmp_path / "source", tmp_path / "export"
    weights = _source(source)
    checkpointer = _Checkpointer(weights, _Adapter())
    hf_export.export_hf(checkpointer, hf_checkpoint=str(source), path=str(destination))
    (destination / "stale.safetensors").write_bytes(b"stale")
    (destination / "sharded").mkdir()
    (destination / "sharded" / "stale.safetensors").write_bytes(b"interrupted save")
    weights["lm_head.weight"].add_(2)
    hf_export.export_hf(checkpointer, hf_checkpoint=str(source), path=str(destination))
    assert not (destination / "stale.safetensors").exists()
    torch.testing.assert_close(_load_export(destination)["lm_head.weight"], weights["lm_head.weight"])


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("sharded", [False, True], ids=["single-file", "multiple-files"])
def test_titan_checkpointer_snapshot_reloads_in_transformers(tmp_path, process_group, tied, sharded):
    titan_checkpoint = pytest.importorskip("torchtitan.components.checkpoint")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    source, destination = tmp_path / "source", tmp_path / "export"
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=tied,
        )
    ).eval()
    model.save_pretrained(source, max_shard_size="1KB" if sharded else "1GB")
    with torch.no_grad():
        model.model.embed_tokens.weight[1].add_(0.1)
    adapter = _Adapter(tied=tied)
    adapter.fqn_to_index_mapping = {name: i % 2 + 1 for i, name in enumerate(model.state_dict())} if sharded else None
    checkpointer = _Checkpointer(model.state_dict(), adapter)
    checkpointer.dcp_save = partial(titan_checkpoint.CheckpointManager.dcp_save, checkpointer)
    hf_export.export_hf(checkpointer, hf_checkpoint=str(source), path=str(destination))

    exported = _load_export(destination)
    torch.testing.assert_close(exported["model.embed_tokens.weight"], model.model.embed_tokens.weight)
    restored = Qwen3ForCausalLM.from_pretrained(destination).eval()
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)


def test_source_checkpoint_cannot_be_overwritten(tmp_path, process_group):
    source = tmp_path / "source"
    weights = _source(source)
    with pytest.raises(ValueError, match="separate"):
        hf_export.export_hf(_Checkpointer(weights, _Adapter()), hf_checkpoint=str(source), path=str(source))
    assert (source / ".complete").exists()


def _distributed_export(rank, root, layout):
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        weights = safetensors.torch.load_file(root / "source" / "model-00001-of-00001.safetensors")
        if layout == "pipeline":
            weights = {name: tensor for i, (name, tensor) in enumerate(sorted(weights.items())) if i % 2 == rank}
        elif layout == "shard":
            mesh = init_device_mesh("cpu", (2,))
            weights = {name: distribute_tensor(tensor, mesh, [Shard(0)]) for name, tensor in weights.items()}
        hf_export.export_hf(
            _Checkpointer(weights, _Adapter()), hf_checkpoint=str(root / "source"), path=str(root / "export")
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["pipeline", "shard"])
def test_collective_export_handles_rank_local_state(tmp_path, layout):
    original = _source(tmp_path / "source")
    mp.spawn(_distributed_export, args=(str(tmp_path), layout), nprocs=2, join=True)
    for name, tensor in _load_export(tmp_path / "export").items():
        torch.testing.assert_close(tensor, original[name])
