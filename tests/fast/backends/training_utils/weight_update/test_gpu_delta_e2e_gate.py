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
    for key in ("WEIGHT_DELTA_CODEC",):
        monkeypatch.delenv(key, raising=False)


def _write_series(tmp_path, changed_bytes, *, protocol=4, codec="snappy-zstd"):
    for version, count in enumerate(changed_bytes, 1):
        directory = tmp_path / f"weight_v{version:06d}"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol_version": protocol,
                    "codec": codec,
                    "stream_id": "current-stream",
                    "base_version": version - 1,
                    "target_version": version,
                    "tensors": [
                        {"name": "w", "shape": [2, 8], "encoding": "xor_bytes", "changed_bytes": count},
                        {"name": "scale", "shape": [], "encoding": "raw_bytes", "changed_bytes": 0, "frames": [], "nbytes": 4},
                    ],
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


def test_e2e_forwards_the_sole_codec_to_ray(monkeypatch):
    assert _gpu_delta_env() == {"WEIGHT_DELTA_CODEC": "snappy-zstd"}
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", "unsupported")
    with pytest.raises(ValueError, match="WEIGHT_DELTA_CODEC"):
        _gpu_delta_env()


@pytest.mark.parametrize("protocol,codec,error", [(3, "snappy-zstd", "protocol"), (4, "zstd", "codec")])
def test_e2e_rejects_a_different_codec(tmp_path, protocol, codec, error):
    final = _write_series(tmp_path, [0, 7, 0], protocol=protocol, codec=codec)
    with patch("torch.distributed.get_rank", return_value=0), pytest.raises(AssertionError, match=error):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])


def test_e2e_rejects_a_compressed_scalar_even_when_weights_changed(tmp_path):
    final = _write_series(tmp_path, [5, 8, 9])
    path = tmp_path / "weight_v000002/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tensors"][1]["encoding"] = "xor_bytes"
    path.write_text(json.dumps(manifest))
    with patch("torch.distributed.get_rank", return_value=0), pytest.raises(AssertionError, match="encoding differs"):
        _assert_gpu_delta_weights_changed(Namespace(num_rollout=4), final, [])
