from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols import nvfp4_gpu


def _family(prefix="model.layers.0.mlp.experts.0.gate_proj"):
    return [
        (prefix + ".weight", torch.zeros((2, 4), dtype=torch.uint8)),
        (prefix + ".weight_scale", torch.zeros((2, 1), dtype=torch.float8_e4m3fn)),
        (prefix + ".weight_scale_2", torch.ones((), dtype=torch.float32)),
    ]


def _manager():
    manager = nvfp4_gpu.Nvfp4GpuDelta.__new__(nvfp4_gpu.Nvfp4GpuDelta)
    manager.error = None
    manager._finished = False
    manager._units = {}
    manager._seen = set()
    manager._pending = deque()
    manager._slots = deque()
    manager._prefetched = {}
    manager._failure_keepalive = []
    manager._capture = False
    manager._result = nvfp4_gpu.GpuDeltaResult()
    return manager


def test_selection_keeps_exclusions_and_shared_experts_ordinary():
    handled = _family()
    ordinary = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(8, dtype=torch.bfloat16)),
        ("model.layers.0.mlp.experts.0.gate_proj.input_scale", torch.ones((), dtype=torch.float32)),
        *_family("model.layers.0.mlp.shared_experts.experts.0.gate_proj"),
        *_family("model.layers.0.self_attn.q_proj"),
    ]
    selected, remaining = nvfp4_gpu._nvfp4_families(handled + ordinary)
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
    manager._layout = MagicMock(side_effect=OSError("GPU layout read failed"))
    family = _family()
    assert manager.process("first", family) == []
    assert manager.process("second", family) == []
    assert isinstance(manager.error, OSError)
    manager._layout.assert_called_once()


def test_canonical_layout_checks_shapes_dtypes_and_contiguity(tmp_path):
    tensors = _family()
    safetensors.torch.save_file(dict(tensors), tmp_path / "model.safetensors")
    manager = _manager()
    # Exercise metadata validation without constructing a CUDA manager.
    manager.device = torch.device("cpu")
    manager.hf_checkpoint = str(tmp_path)

    regions = manager._layout(tensors)
    assert [region.offset for region in regions] == [0, 8, 10]
    assert [region.nbytes for region in regions] == [8, 2, 4]
    with pytest.raises(ValueError, match="Canonical NVFP4 layout differs"):
        manager._layout([(tensors[0][0], torch.zeros((2, 4), dtype=torch.float32)), *tensors[1:]])
    with pytest.raises(ValueError, match="Canonical NVFP4 layout differs"):
        manager._layout([(tensors[0][0], torch.zeros((4, 2), dtype=torch.uint8)), *tensors[1:]])
    with pytest.raises(ValueError, match="contiguous canonical storage"):
        manager._layout([(tensors[0][0], torch.zeros((4, 2), dtype=torch.uint8).t()), *tensors[1:]])
    manager._units["unit"] = SimpleNamespace(regions=regions)
    assert manager.process("unit", list(reversed(tensors))) == []
    assert isinstance(manager.error, ValueError) and "layout changed" in str(manager.error)


@pytest.mark.parametrize("failure", ["encode", "close", "writeback"])
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

    pending = nvfp4_gpu._Pending(
        unit=SimpleNamespace(regions=()),
        slot=(object(), object()),
        encoded=SimpleNamespace(finish=operation("encode", []), close=operation("close")),
        written=SimpleNamespace(synchronize=operation("writeback")),
        inputs=(torch.zeros(1),),
    )
    manager._pending.append(pending)
    with pytest.raises(OSError) as error:
        manager._collect_one()
    assert error.value is original
    assert events == ["encode", "close", "writeback"]
    assert not manager._slots
    assert manager._failure_keepalive == [pending]


@pytest.mark.parametrize("stream_fails", [False, True])
def test_finish_drains_prefetches_and_streams_after_one_failure(stream_fails):
    manager = _manager()
    original = OSError("prefetch failed")
    first, second = MagicMock(), MagicMock()
    first.synchronize.side_effect = original
    manager._prefetched = {"first": (object(), first), "second": (object(), second)}
    manager._producer_streams = {MagicMock()}
    manager._prefetch_stream, manager._codec_stream, manager._writeback_stream = MagicMock(), MagicMock(), MagicMock()

    retained = object()
    manager._failure_keepalive = [retained]
    if stream_fails:
        manager._prefetch_stream.synchronize.side_effect = OSError("stream failed")

    with pytest.raises(RuntimeError, match="failed") as error:
        manager.finish()

    assert error.value.__cause__ is original
    second.synchronize.assert_called_once_with()
    for stream in (
        *manager._producer_streams,
        manager._prefetch_stream,
        manager._codec_stream,
        manager._writeback_stream,
    ):
        stream.synchronize.assert_called_once_with()
    assert bool(manager._prefetched) == stream_fails
    assert manager._failure_keepalive == ([retained] if stream_fails else [])
