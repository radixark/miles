"""Startup checkpoint version declaration must precede the first rollout."""

import asyncio
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import safetensors.numpy
import torch

from miles.backends.training_utils.weight_update.protocols import gpu_delta


class _Engine:
    def __init__(self, protocol, events, index, fail=False):
        self.protocol, self.events, self.index, self.fail = protocol, events, index, fail
        self.version = "default"

    async def update_weight_version(self, *, weight_version):
        assert not self.protocol._baseline_captured
        np.testing.assert_array_equal(self.protocol._snapshot["w"], [1, 2, 3, 4])
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append(self.index)
        if self.fail:
            raise RuntimeError("engine rejected base declaration")
        self.version = weight_version
        return {"success": True, "new_version": weight_version}


def _setup(tmp_path, *, fail=False):
    safetensors.numpy.save_file({"w": np.array([1, 2, 3, 4], dtype=np.uint8)}, tmp_path / "model.safetensors")
    protocol = gpu_delta.UpdateWeightFromGpuDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            custom_update_weight_post_write_path=None,
        )
    )
    protocol._plan = {"w": {"name": "w", "dtype": "U8", "shape": [4]}}
    protocol.is_sender = True
    events = []
    protocol.rollout_engines = [_Engine(protocol, events, 0, fail), _Engine(protocol, events, 1)]
    return protocol, events


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", "cpu")
    with (
        patch.object(gpu_delta, "_gather_all", side_effect=lambda value: [value]),
        patch.object(gpu_delta, "get_gloo_group", return_value=None),
        patch.object(gpu_delta.dist, "get_rank", return_value=0),
        patch.object(gpu_delta.dist, "broadcast_object_list"),
    ):
        yield


def _buckets(*, materialize):
    assert materialize
    # The baseline must use the startup checkpoint, not this exported value.
    yield [("w", torch.tensor([5, 6, 7, 8], dtype=torch.uint8))]


def test_verified_startup_checkpoint_is_zero_on_every_engine_before_rollout(tmp_path, single_rank):
    protocol, events = _setup(tmp_path)
    assert protocol.begin_sync(1, _buckets) is False  # no optimizer/update version increment
    assert sorted(events) == [0, 1]
    assert [engine.version for engine in protocol.rollout_engines] == ["0", "0"]
    assert protocol._baseline_captured and not protocol._uncommitted
    assert not protocol._stream_dir.exists()  # no pretend startup publication


def test_partial_version_acknowledgement_fails_before_rollout_without_replay(tmp_path, single_rank):
    protocol, events = _setup(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="engine rejected base declaration"):
        protocol.begin_sync(1, _buckets)
    assert sorted(events) == [0, 1]  # all submitted RPCs settled before broadcasting failure
    assert not protocol._baseline_captured and protocol._uncommitted
    with pytest.raises(RuntimeError, match="automatic replay is forbidden"):
        protocol.begin_sync(1, _buckets)
    assert len(events) == 2


def test_inventory_failure_never_declares_base_version(tmp_path, single_rank):
    protocol, events = _setup(tmp_path)
    protocol._plan["missing"] = {"name": "missing", "dtype": "U8", "shape": [4]}
    with pytest.raises(RuntimeError, match="inventory/ownership mismatch"):
        protocol.begin_sync(1, _buckets)
    assert not events and not protocol._baseline_captured
