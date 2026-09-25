import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize(
    ("overrides", "expected_size"),
    [
        ({"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}, 4),
        ({"hardware": "GB300", "num_nodes": 8, "num_gpus_per_node": 4}, 8),
        (
            {
                "hardware": "GB300",
                "num_nodes": 16,
                "num_gpus_per_node": 4,
                "rollout_num_nodes": 8,
                "update_weight_transfer_mode": "broadcast",
            },
            8,
        ),
        (
            {
                "hardware": "GB300",
                "num_nodes": 16,
                "num_gpus_per_node": 4,
                "rollout_num_nodes": 8,
                "update_weight_transfer_mode": "nccl-m2n",
            },
            8,
        ),
        (
            {
                "hardware": "H200",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
        (
            {
                "hardware": "GB300",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
    ],
)
def test_rollout_and_memory_profiles_follow_the_topology(monkeypatch, tmp_path, overrides, expected_size):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command
    uses_gpu_peak = overrides["hardware"] == "GB300" and overrides.get("rollout_num_nodes", 0) == 0
    assert ("--colocate-memory-peak-device gpu" in train_command) == uses_gpu_peak
