import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.numpy
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _make_protocol(calls: list[tuple[str, dict]]) -> UpdateWeightFromDiskDelta:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode="retract",
        check_weight_update_equal=False,
    )
    protocol.rollout_engines = [_RecordingApiClient(calls)]
    protocol._post_write_hook = None
    protocol._version_dir = "/shared/delta/v7"
    protocol._snapshot_file_version = 7
    return protocol


def test_reload_engines_pulls_with_both_checkpoint_dirs_then_reloads():
    """The reload pull carries both checkpoint dirs the deleted engine wrapper used to inject."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(weight_version=42)

    assert [name for name, _kwargs in calls] == [
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert calls[1][1] == {"mode": "retract"}
    assert calls[0][1] == {
        "target_version": 7,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }
    assert calls[3][1] == {"model_path": "/local/ckpt", "weight_version": "42"}


def test_in_place_pause_mode_skips_the_flush():
    """in_place pause mode does not flush."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)
    protocol.args.pause_generation_mode = "in_place"

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(weight_version=42)

    assert "flush_cache" not in [name for name, _kwargs in calls]


def test_non_source_rank_issues_no_requests():
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 1
        protocol._reload_engines(weight_version=42)

    assert calls == []


def _capture_baseline(protocol: UpdateWeightFromDiskDelta, tmp_path) -> None:
    protocol.delta_dir = str(tmp_path / "delta")
    protocol.args.hf_checkpoint = "/fake/hf"
    protocol._snapshot = {}
    protocol.is_sender = False

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name, **kwargs: None),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        protocol._capture_baseline(lambda materialize: [])


@pytest.mark.parametrize("check_weight_update_equal", [False, True])
def test_baseline_capture_only_materializes_the_checkpoint(tmp_path, check_weight_update_equal):
    """Engine weights and version are updated by the ensuing publication, not baseline capture."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)
    protocol.args.check_weight_update_equal = check_weight_update_equal

    _capture_baseline(protocol, tmp_path)

    assert [name for name, _kwargs in calls] == ["pull_weights"]
    assert calls[0][1] == {
        "target_version": 0,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }


def test_non_source_rank_waits_for_baseline_materialization(tmp_path):
    """Every rank waits until rank zero has checked the baseline pull results."""
    protocol = _make_protocol([])
    protocol.delta_dir = str(tmp_path / "delta")
    protocol.args.hf_checkpoint = "/fake/hf"
    protocol._snapshot = {}
    protocol.is_sender = False

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name, **kwargs: None),
    ):
        dist_mock.get_rank.return_value = 1
        dist_mock.get_world_size.return_value = 1
        protocol._capture_baseline(lambda materialize: [])

    assert dist_mock.barrier.call_count == 2


@pytest.mark.parametrize("changed", [False, True])
def test_initial_sync_encodes_loaded_trainer_from_hf_baseline(tmp_path, changed):
    """The first call publishes and reloads trainer weights, including an empty delta."""
    old = np.array([1, 2, 3, 4], dtype=np.uint8)
    new = np.array([5, 6, 7, 8], dtype=np.uint8) if changed else old.copy()
    safetensors.numpy.save_file({"w": old}, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
            check_weight_update_equal=False,
            pause_generation_mode="retract",
        )
    )
    calls = []
    protocol.rollout_engines = [_RecordingApiClient(calls)]
    protocol.is_sender = True

    def buckets(materialize):
        assert materialize
        yield [("w", torch.from_numpy(new))]

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=None),
        patch.object(protocol, "_record_metrics"),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = lambda values, value, **kwargs: values.__setitem__(0, value)
        assert protocol.begin_sync(37, buckets) is True
        np.testing.assert_array_equal(protocol._snapshot["w"], old)
        protocol._use_pinned = False  # CPU test; production's CUDA staging is covered separately.
        for bucket in buckets(materialize=True):
            protocol.send_bucket(bucket)
        protocol.after_base_weights()
        protocol.finalize(37)
    version_dir = tmp_path / "delta" / "weight_v000001"
    index = json.loads((version_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["base_version"] == "000000"
    if changed:
        delta = safetensors.numpy.load_file(version_dir / index["weight_map"]["w"])
        decoded = np.frombuffer(zstandard.ZstdDecompressor().decompress(delta["w"].tobytes()), dtype=np.uint8)
        np.testing.assert_array_equal(old ^ decoded, new)
    else:
        assert index["weight_map"] == {}
    assert [name for name, _kwargs in calls] == [
        "pull_weights",
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert [calls[i][1]["target_version"] for i in (0, 1)] == [0, 1]
    assert calls[4][1] == {"model_path": "/local/ckpt", "weight_version": "37"}
    np.testing.assert_array_equal(protocol._snapshot["w"], new)
