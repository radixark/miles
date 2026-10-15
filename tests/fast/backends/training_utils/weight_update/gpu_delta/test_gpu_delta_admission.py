"""GPU-delta startup requires a real immutable receiver baseline."""

from argparse import Namespace

import pytest

from miles.utils.gpu_delta.admission import validate_gpu_delta_args


def _args(tmp_path, **overrides):
    return Namespace(
        **(
            {
                "update_weight_transfer_mode": "gpu-delta",
                "update_weight_delta_initial_sync": False,
                "update_weight_delta_encoding": "xor",
                "colocate": False,
                "train_backend": "megatron",
                "megatron_to_hf_mode": "raw",
                "lora_rank": 0,
                "prefill_num_servers": None,
                "pause_generation_mode": "retract",
                "check_weight_update_equal": False,
                "update_weight_disk_dir": str(tmp_path / "updates"),
                "hf_checkpoint": str(tmp_path),
                "mtp_num_layers": None,
                "sglang_load_format": "auto",
                "sglang_speculative_draft_load_format": None,
            }
            | overrides
        )
    )


def test_gpu_delta_requires_initial_sync_but_other_protocols_keep_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_DELTA_CODEC", "lz4")
    monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", "snappy-zstd")
    args = _args(tmp_path)
    validate_gpu_delta_args(args)
    assert args.update_weight_delta_initial_sync is True
    disk = _args(tmp_path, update_weight_transfer_mode="disk-delta")
    validate_gpu_delta_args(disk)
    assert disk.update_weight_delta_initial_sync is False


@pytest.mark.parametrize(
    "field",
    [
        "sglang_load_format",
        "sglang_speculative_draft_load_format",
        "eval_sglang_load_format",
        "eval_sglang_speculative_draft_load_format",
    ],
)
def test_dummy_target_or_draft_cannot_establish_delta_baseline(tmp_path, field):
    with pytest.raises(ValueError, match="checkpoint|safetensors"):
        validate_gpu_delta_args(_args(tmp_path, **{field: "dummy"}))


def test_trainable_draft_and_invalid_initial_codec_fail_before_startup(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="static receiver MTP"):
        validate_gpu_delta_args(_args(tmp_path, mtp_num_layers=1))
    monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", "invalid")
    with pytest.raises(ValueError, match="CODEC"):
        validate_gpu_delta_args(_args(tmp_path))
