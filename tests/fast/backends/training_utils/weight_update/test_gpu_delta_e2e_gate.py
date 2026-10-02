"""The short GPU-delta E2E must not pass on version-only publications."""

import json
from argparse import Namespace
from unittest.mock import patch

import pytest
from tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16 import (
    _assert_gpu_delta_weights_changed,
    _gpu_delta_env,
)


@pytest.fixture(autouse=True)
def _default_delta_env(monkeypatch):
    for key in ("WEIGHT_DELTA_CODEC", "WEIGHT_DELTA_ENCODER", "WEIGHT_DELTA_SNAPPY_ZSTD"):
        monkeypatch.delenv(key, raising=False)


def _write_series(tmp_path, changed_bytes, *, protocol=2, profile="snappy-independent-1mib-v1"):
    for version, count in enumerate(changed_bytes, 1):
        directory = tmp_path / f"weight_v{version:06d}"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol_version": protocol,
                    "codec_profile": profile,
                    "stream_id": "current-stream",
                    "base_version": version - 1,
                    "target_version": version,
                    "tensors": [{"changed_bytes": count}],
                }
            )
        )
    return directory


@pytest.mark.parametrize("changed_bytes", [[0, 7, 0], [5, 8, 9]])
def test_complete_series_with_actual_changed_bytes_is_accepted(tmp_path, changed_bytes):
    final = _write_series(tmp_path, changed_bytes)
    with patch("torch.distributed.get_rank", return_value=0):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])


def test_three_noop_updates_cannot_pass_the_e2e(tmp_path):
    final = _write_series(tmp_path, [0, 0, 0])
    with patch("torch.distributed.get_rank", return_value=0), pytest.raises(AssertionError, match="only no-op"):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])


def test_changed_bytes_from_another_stream_do_not_count(tmp_path):
    final = _write_series(tmp_path, [0, 7, 0])
    path = tmp_path / "weight_v000002/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stream_id"] = "other-stream"
    path.write_text(json.dumps(manifest))
    with patch("torch.distributed.get_rank", return_value=0), pytest.raises(AssertionError):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])


@pytest.mark.parametrize("wrapped", ["0", "1"])
def test_e2e_explicitly_forwards_outer_encoding_mode_to_ray(monkeypatch, wrapped):
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", wrapped)
    assert _gpu_delta_env() == {
        "WEIGHT_DELTA_CODEC": "snappy",
        "WEIGHT_DELTA_ENCODER": "gpu",
        "WEIGHT_DELTA_SNAPPY_ZSTD": wrapped,
    }


def test_outer_zstd_series_is_accepted_when_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", "1")
    final = _write_series(tmp_path, [0, 7, 0], protocol=3, profile="snappy-independent-1mib-zstd-v1")
    with patch("torch.distributed.get_rank", return_value=0):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])


@pytest.mark.parametrize(
    ("protocol", "profile", "error"),
    [(2, "snappy-independent-1mib-v1", "protocol"), (3, "snappy-independent-1mib-v1", "codec profile")],
)
def test_outer_zstd_request_rejects_an_unwrapped_publication(tmp_path, monkeypatch, protocol, profile, error):
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", "1")
    final = _write_series(tmp_path, [0, 7, 0], protocol=protocol, profile=profile)
    with patch("torch.distributed.get_rank", return_value=0), pytest.raises(AssertionError, match=error):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])
