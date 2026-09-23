import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors.torch
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


@pytest.fixture
def exporter(monkeypatch, tmp_path):
    class WeightUpdatePlacement:
        def __init__(self, *, gather_pp):
            self.gather_pp = gather_pp

    stubs = {
        "megatron.core.distributed": {"DistributedDataParallel": object},
        "miles.backends.megatron_utils.lora.utils": {
            "is_lora_model": lambda model: False,
            "save_lora_checkpoint": lambda *args: None,
        },
        "miles.backends.megatron_utils.named_weights": {
            "named_params_and_buffers": lambda *args, **kwargs: [],
        },
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct": {
            "HfWeightIteratorDirect": object,
        },
        "miles.backends.training_utils.weight_update.hf_weight_iterator": {
            "WeightUpdatePlacement": WeightUpdatePlacement,
        },
        "miles.backends.training_utils.weight_update.utils": {
            "get_data_replica_rank_and_size": lambda parallel_state, placement: (0, 1),
        },
        "miles.utils.hf_config": {
            "HF_EXPORT_COMPLETE_MARKER": ".complete",
            "load_hf_config": lambda path: SimpleNamespace(quantization_config=None),
        },
        "miles.utils.megatron_bridge_utils": {"patch_megatron_model": lambda model: nullcontext()},
        "miles.utils.distributed_utils": {"get_gloo_group": lambda: None},
        "miles.backends.training_utils.parallel": {
            "get_parallel_state": lambda: SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=0)),
        },
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/hf_export.py"
    spec = importlib.util.spec_from_file_location("hf_export_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    args = SimpleNamespace(
        hf_checkpoint=str(base),
        save_hf=str(tmp_path / "export"),
        save_hf_writers=1,
        model_name=None,
        megatron_to_hf_mode="raw",
    )
    chunks = [[("first", torch.arange(4))], [("second", torch.ones(4))]]
    monkeypatch.setattr(
        module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(iter_hf_weights=lambda weights: iter(chunks)),
    )
    rank = threading.local()
    rank.value = 0
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: getattr(rank, "value", 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda output, value, **kwargs: output.__setitem__(0, value))
    return SimpleNamespace(module=module, args=args, path=Path(args.save_hf), chunks=chunks, rank=rank)


def test_save_hf_writes_loadable_checkpoint_and_complete_marker(exporter):
    exporter.module.save_hf_model(exporter.args, 3, [])

    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 48
    assert (exporter.path / ".complete").exists()
    assert (exporter.path / "config.json").exists()
    assert torch.equal(
        safetensors.torch.load_file(exporter.path / index["weight_map"]["first"])["first"],
        torch.arange(4),
    )


def test_source_completion_marker_is_not_metadata(exporter):
    source_marker = Path(exporter.args.hf_checkpoint) / ".complete"
    source_marker.touch()

    assert not exporter.module._is_hf_metadata_file(source_marker)


def test_shard_writer_bounds_each_rank_to_one_pending_shard(exporter, monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    original_save = safetensors.torch.save_file
    calls = 0

    def save(tensors, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert release_first.wait(5)
        original_save(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", save)
    exporter.path.mkdir()
    writer = exporter.module._AsyncShardWriter(exporter.path)
    writer.submit("first.safetensors", {"first": torch.arange(4)})
    assert first_started.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        second = pool.submit(writer.submit, "second.safetensors", {"second": torch.ones(4)})
        assert not second.done()
        release_first.set()
        second.result(timeout=5)
    writer.finish()

    assert calls == 2


def test_multiple_writers_publish_distinct_shards(exporter, monkeypatch):
    world_size = 4
    rendezvous = threading.Barrier(world_size, timeout=10)
    gathered = [None] * world_size
    expected = {f"weight_{index}": torch.arange(8, dtype=torch.float32) + index for index in range(4)}

    def all_gather(output, value, **kwargs):
        gathered[exporter.rank.value] = value
        rendezvous.wait()
        output[:] = gathered
        rendezvous.wait()

    def chunks():
        for name, tensor in expected.items():
            # Full placement makes every data replica expose the same canonical
            # HF tensors, independent of its original TP/EP/PP coordinates.
            yield [(name, tensor)]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args, **kwargs: world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather)
    monkeypatch.setattr(
        exporter.module,
        "get_parallel_state",
        lambda: SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=exporter.rank.value % 2)),
    )
    monkeypatch.setattr(
        exporter.module,
        "get_data_replica_rank_and_size",
        lambda parallel_state, placement: (exporter.rank.value, world_size),
    )
    monkeypatch.setattr(
        exporter.module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(iter_hf_weights=lambda weights: chunks()),
    )
    exporter.args.save_hf_writers = 2

    def run(rank):
        exporter.rank.value = rank
        exporter.module.export_hf_model_direct(
            exporter.args,
            [],
            exporter.path,
            model_name="model",
            quantization_config=None,
            megatron_local_weights={},
        )

    with ThreadPoolExecutor(world_size) as pool:
        futures = [pool.submit(run, rank) for rank in range(world_size)]
        for future in futures:
            future.result(timeout=10)

    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 4
    for name, tensor in expected.items():
        saved = safetensors.torch.load_file(exporter.path / index["weight_map"][name])[name]
        assert torch.equal(saved, tensor), name


def test_writer_failure_is_collective_and_does_not_commit(exporter, monkeypatch):
    monkeypatch.setattr(safetensors.torch, "save_file", lambda *args: (_ for _ in ()).throw(OSError("full")))

    with pytest.raises(RuntimeError, match="OSError: full"):
        exporter.module.save_hf_model(exporter.args, 3, [], raise_on_error=True)

    assert not (exporter.path / "model.safetensors.index.json").exists()
    assert not (exporter.path / ".complete").exists()


def test_writer_count_cannot_exceed_complete_model_replicas(exporter):
    exporter.args.save_hf_writers = 2

    with pytest.raises(ValueError, match="exceeds the 1 complete model replicas"):
        exporter.module.export_hf_model_direct(
            exporter.args,
            [],
            exporter.path,
            model_name="model",
            quantization_config=None,
            megatron_local_weights={},
        )
