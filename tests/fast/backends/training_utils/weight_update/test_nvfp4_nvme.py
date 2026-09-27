from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols import nvfp4_nvme


def _family(prefix="model.layers.0.mlp.experts.0.gate_proj"):
    return [
        (prefix + ".weight", torch.zeros((2, 4), dtype=torch.uint8)),
        (prefix + ".weight_scale", torch.zeros((2, 1), dtype=torch.float8_e4m3fn)),
        (prefix + ".weight_scale_2", torch.ones((), dtype=torch.float32)),
    ]


def _manager():
    manager = nvfp4_nvme.Nvfp4NvmeDelta.__new__(nvfp4_nvme.Nvfp4NvmeDelta)
    manager.error = None
    manager._units = {}
    manager._seen = set()
    manager._pending = deque()
    manager._slots = deque()
    manager._prefetched = {}
    manager._failure_keepalive = []
    manager._capture = False
    manager._reader = manager._writer = None
    manager._result = nvfp4_nvme.NvmeDeltaResult()
    return manager


def test_selection_keeps_exclusions_and_shared_experts_ordinary():
    handled = _family()
    ordinary = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(8, dtype=torch.bfloat16)),
        *_family("model.layers.0.mlp.shared_experts.experts.0.gate_proj"),
        *_family("model.layers.0.self_attn.q_proj"),
    ]
    selected, remaining = nvfp4_nvme._nvfp4_families(handled + ordinary)
    assert [name for name, _ in selected] == [name for name, _ in handled]
    assert [name for name, _ in remaining] == [name for name, _ in ordinary]


@pytest.mark.parametrize("corruption", ["weight_dtype", "missing_scale", "scale_dtype"])
def test_malformed_previous_family_never_reenters_cached_gather(corruption):
    manager = _manager()
    handled = _family()
    manager._units["unit"] = SimpleNamespace(regions=[SimpleNamespace(name=name) for name, _ in handled])
    if corruption == "weight_dtype":
        handled[0] = (handled[0][0], handled[0][1].to(torch.bfloat16))
    elif corruption == "missing_scale":
        handled.pop(1)
    else:
        handled[1] = (handled[1][0], handled[1][1].to(torch.float32))
    ordinary = ("model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(8, dtype=torch.bfloat16))

    remaining = manager.process("unit", handled + [ordinary])

    assert remaining == [ordinary]
    assert isinstance(manager.error, ValueError)


def test_backend_failure_keeps_consuming_handled_families():
    manager = _manager()
    manager._layout = MagicMock(side_effect=OSError("NVMe layout read failed"))
    family = _family()
    assert manager.process("first", family) == []
    assert manager.process("second", family) == []
    assert isinstance(manager.error, OSError)
    manager._layout.assert_called_once()


def test_canonical_layout_checks_shapes_dtypes_order_and_contiguity(tmp_path, monkeypatch):
    tensors = _family()
    safetensors.torch.save_file(dict(tensors), tmp_path / "model.safetensors")
    manager = _manager()
    # This exercises layout metadata only; real construction still requires CUDA.
    manager.device = torch.device("cpu")
    manager.hf_checkpoint = str(tmp_path)
    manager._capture = True
    manager._size = 0
    manager._next_path = tmp_path / "next.bin"
    manager._next_path.touch()
    monkeypatch.setattr(nvfp4_nvme.os, "posix_fallocate", lambda *args: None, raising=False)

    unit = manager._layout("unit", tensors)
    assert unit.nbytes == 4096
    assert [region.offset for region in unit.regions] == [0, 8, 10]
    assert [region.nbytes for region in unit.regions] == [8, 2, 4]
    assert manager._layout("unit", tensors) is unit
    with pytest.raises(ValueError, match="layout changed"):
        manager._layout("unit", list(reversed(tensors)))
    with pytest.raises(ValueError, match="Canonical NVFP4 layout differs"):
        manager._layout("unit", [(tensors[0][0], torch.zeros((2, 4), dtype=torch.float32)), *tensors[1:]])
    with pytest.raises(ValueError, match="Canonical NVFP4 layout differs"):
        manager._layout("unit", [(tensors[0][0], torch.zeros((4, 2), dtype=torch.uint8)), *tensors[1:]])
    with pytest.raises(ValueError, match="contiguous canonical NVFP4"):
        manager._layout("unit", [(tensors[0][0], torch.zeros((4, 2), dtype=torch.uint8).t()), *tensors[1:]])


@pytest.mark.parametrize("failure", ["encode", "close", "write"])
def test_failed_pending_batch_drains_both_consumers_before_releasing_storage(failure):
    manager = _manager()
    events = []
    original = OSError(f"{failure} failed")

    def operation(name, result=None):
        def run():
            events.append(name)
            if name == failure:
                raise original
            return result

        return run

    pending = nvfp4_nvme._Pending(
        unit=SimpleNamespace(regions=()),
        slot=object(),
        write=SimpleNamespace(result=operation("write")),
        encoded=SimpleNamespace(finish=operation("encode", []), close=operation("close")),
        inputs=(torch.zeros(1),),
    )
    manager._pending.append(pending)
    with pytest.raises(OSError) as error:
        manager._collect_one()
    assert error.value is original
    assert events == ["encode", "close", "write"]
    assert not manager._slots
    assert manager._failure_keepalive == [pending]


def test_finish_drains_prefetches_and_both_handles_after_one_failure(monkeypatch):
    manager = _manager()
    original = OSError("read failed")
    first, second = MagicMock(), MagicMock()
    first.result.side_effect = original
    manager._prefetched = {"first": (object(), first), "second": (object(), second)}
    reader, writer = MagicMock(), MagicMock()
    reader.close.side_effect = OSError("close failed")
    manager._reader, manager._writer = reader, writer
    manager.device = torch.device("cuda", 0)
    manager._stream = MagicMock()
    current = MagicMock()
    monkeypatch.setattr(nvfp4_nvme.torch.cuda, "current_stream", lambda device: current)

    with pytest.raises(RuntimeError, match="preparation failed") as error:
        manager.finish()

    assert error.value.__cause__ is original
    second.result.assert_called_once_with()
    reader.close.assert_called_once_with()
    writer.close.assert_called_once_with()
    current.synchronize.assert_called_once_with()
    manager._stream.synchronize.assert_called_once_with()
    assert not manager._prefetched
    assert manager._reader is manager._writer is None


def test_begin_failure_closes_open_handles_and_poisoned_state_cannot_restart(tmp_path, monkeypatch):
    manager = _manager()
    manager.device = torch.device("cpu")
    manager.directory = tmp_path
    manager._version = 0
    manager._size = 4096
    manager._units = {"unit": SimpleNamespace(nbytes=4096)}
    manager._read_executor = object()
    manager._write_executor = object()
    writer = MagicMock()
    original = OSError("reader open failed")
    manager._backend = MagicMock(side_effect=[writer, original])
    writer.close.side_effect = OSError("writer close failed")
    monkeypatch.setattr(nvfp4_nvme.os, "posix_fallocate", lambda *args: None, raising=False)

    with pytest.raises(OSError) as error:
        manager.begin(capture_baseline=False, weight_version=1)

    assert error.value is original
    writer.close.assert_called_once_with()
    assert manager._reader is manager._writer is None
    assert manager.error is original
    with pytest.raises(RuntimeError, match="failed NVMe baseline cannot be reused"):
        manager.begin(capture_baseline=False, weight_version=1)
