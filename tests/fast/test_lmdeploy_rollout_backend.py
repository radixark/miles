import shlex
from types import SimpleNamespace

import pytest

from miles.backends.lmdeploy_utils.launch import compute_lmdeploy_launch_cmd, validate_lmdeploy_args


def _config(tmp_path, **updates):
    python = tmp_path / "serving python"
    python.touch()
    values = dict(
        rollout_engine_backend="lmdeploy",
        lmdeploy_python=str(python),
        router="miles",
        colocate=True,
        rollout_num_gpus_per_engine=1,
        lmdeploy_max_batch_size=64,
        fully_async=False,
        partial_rollout=False,
        rollout_external=False,
        lora_rank=0,
        actor_num_nodes=1,
        pause_generation_mode="retract",
        check_weight_update_equal=False,
        colocate_memory_peak_device="cpu",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_session_server=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang=SimpleNamespace(base_args={"speculative_algorithm": None, "api_key": None}),
    )
    return SimpleNamespace(**(values | updates))


@pytest.mark.parametrize("feature", ["fully_async", "partial_rollout", "check_weight_update_equal"])
def test_unsupported_training_modes_fail_before_starting_workers(tmp_path, feature):
    with pytest.raises(ValueError, match="does not support"):
        validate_lmdeploy_args(_config(tmp_path, **{feature: True}))


def test_single_gpu_launch_preserves_visible_device_mapping_and_environment_isolation(tmp_path, monkeypatch):
    args = _config(tmp_path)
    validate_lmdeploy_args(args)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    command = shlex.split(
        compute_lmdeploy_launch_cmd(
            args=args,
            model_path="/model with spaces",
            host="127.0.0.1",
            port=9001,
            gate_port=9002,
            base_gpu_id=2,
            cache_fraction=0.8,
        )
    )
    assert command[:5] == ["env", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONPATH"]
    assert "CUDA_VISIBLE_DEVICES=6" in command
    assert command[command.index("--model") + 1] == "/model with spaces"
    assert command[command.index("--gate-port") + 1] == "9002"
    assert command[command.index("--max-batch-size") + 1] == "64"


def test_sglang_does_not_require_a_serving_python(tmp_path):
    validate_lmdeploy_args(_config(tmp_path, rollout_engine_backend="sglang", lmdeploy_python=None))


def test_portable_bucket_keeps_a_snapshot_until_receiver_acknowledgement():
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    from miles.backends.lmdeploy_utils.tensor_buckets import serialize_tensor_bucket

    weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    expected = weight.clone()
    payload, lifetime = serialize_tensor_bucket([("model.norm.weight", weight)])
    path = Path(payload[0])
    try:
        weight.zero_()
        torch.testing.assert_close(load_file(str(path))["model.norm.weight"], expected)
        assert path.exists()
    finally:
        lifetime.cleanup()
    assert not path.exists()
