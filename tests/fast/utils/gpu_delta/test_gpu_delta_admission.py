"""GPU-delta baseline and launch-argument admission."""

from argparse import Namespace
from contextlib import nullcontext

import pytest

from miles.utils.gpu_delta.admission import validate_gpu_delta_args


def _args(tmp_path, **overrides):
    return Namespace(
        **(
            {
                "update_weight_transfer_mode": "gpu-delta",
                "update_weight_delta_encoding": "xor",
                "colocate": False,
                "offload_rollout": None,
                "debug_train_only": False,
                "eval_num_gpus": 0,
                "ft_components": [],
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


@pytest.mark.parametrize("mode,admitted", [("retract", True), ("in_place", True), ("abort", False)])
def test_receiver_pause_modes_are_admitted_and_abort_is_rejected(tmp_path, mode, admitted):
    with nullcontext() if admitted else pytest.raises(AssertionError, match="retract or in_place"):
        validate_gpu_delta_args(_args(tmp_path, pause_generation_mode=mode))


def test_unsupported_launch_configuration_fails_before_startup(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="snapshot publisher"):
        validate_gpu_delta_args(_args(tmp_path, debug_train_only=True, eval_num_gpus=2))
    with pytest.raises(AssertionError, match="sender baseline loss"):
        validate_gpu_delta_args(_args(tmp_path, ft_components=["train"]))
    with pytest.raises(AssertionError, match="static receiver MTP"):
        validate_gpu_delta_args(_args(tmp_path, mtp_num_layers=1))
    with pytest.raises(AssertionError, match="resident original rollout weight"):
        validate_gpu_delta_args(_args(tmp_path, offload_rollout=True))
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "yes")
    with pytest.raises(ValueError, match="SKIP_PAYLOAD_HASH"):
        validate_gpu_delta_args(_args(tmp_path))
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "1")
    monkeypatch.setenv("GPU_DELTA_INITIAL_SYNC_CODEC", "invalid")
    with pytest.raises(ValueError, match="CODEC"):
        validate_gpu_delta_args(_args(tmp_path))
