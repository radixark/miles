"""The short GPU-delta E2E must not pass on version-only publications."""

import json
import os
import shlex
from argparse import Namespace
from unittest.mock import patch

import pytest
from tests.e2e.megatron import test_glm5_2_744b_a40b_5layer_nvfp4_w4a16 as e2e
from tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16 import (
    _assert_gpu_delta_weights_changed,
    _gpu_delta_env,
)


@pytest.fixture(autouse=True)
def _default_delta_env(monkeypatch):
    monkeypatch.delenv("WEIGHT_DELTA_CODEC", raising=False)


def _write_series(tmp_path, changed_bytes, protocol=4, codec="snappy-zstd"):
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
    assert _gpu_delta_env() == {"WEIGHT_DELTA_CODEC": "snappy-zstd", "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN": "0"}
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", "unsupported")
    with pytest.raises(ValueError, match="WEIGHT_DELTA_CODEC"):
        _gpu_delta_env()


@pytest.mark.parametrize("gpu_delta", [False, True])
def test_gpu_delta_is_explicit_and_preserves_the_ordinary_e2e_arm(monkeypatch, gpu_delta):
    launch = {}
    backend = Namespace(execute_train=lambda **kwargs: launch.update(kwargs))
    monkeypatch.setattr(e2e.command_utils, "default_config", lambda: Namespace(create_backend=lambda: backend))
    monkeypatch.setattr(e2e.command_utils, "encode_pseudo_file", lambda _: "/tmp/precision.yaml")
    monkeypatch.setattr(e2e.command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")
    with patch.dict(os.environ):
        e2e.execute(**({"gpu_delta": True, "update_weight_disk_dir": "/tmp/delta outputs"} if gpu_delta else {}))
    args = shlex.split(launch["train_args"])
    assert args[args.index("--update-weight-transfer-mode") + 1] == ("gpu-delta" if gpu_delta else "broadcast_packed")
    assert args[args.index("--rm-type") + 1] == ("deterministic_random" if gpu_delta else "deepscaler")
    assert args[args.index("--num-rollout") + 1] == ("4" if gpu_delta else "2")
    assert ("--use-fault-tolerance" in args) is not gpu_delta
    assert ("--custom-update-weight-post-write-path" in args) is gpu_delta
    assert ("WEIGHT_DELTA_CODEC" in launch["extra_env_vars"]) is gpu_delta
    assert ("SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN" in launch["extra_env_vars"]) is gpu_delta
    if gpu_delta:
        assert args[args.index("--update-weight-disk-dir") + 1] == "/tmp/delta outputs"


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
