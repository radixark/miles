from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


def test_qwen36_sft_profile_pins_model_data_and_observability(monkeypatch, tmp_path) -> None:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_sft.py")

    call_entrypoint(
        module,
        "execute",
        {
            "model_name": "Qwen3.6-35B-A3B",
            "run_id": "260827-12345678",
            "prompt_data": "/datasets/sft.parquet",
            "checkpoint_dir": "/checkpoints/260827-12345678",
            "trace_dir": "/scratch/260827-12345678/traces",
            "tools_key": "tools",
            "metadata_key": "meta",
            "tensor_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "max_tokens_per_gpu": 262144,
            "wandb_project": "traffic-sft-smoke",
        },
        sandbox=tmp_path,
    )

    train_command = recording.commands[-1]
    expected_fragments = (
        "--hf-checkpoint /root/models/Qwen3.6-35B-A3B",
        "--ref-load /root/models/Qwen3.6-35B-A3B_torch_dist",
        "--prompt-data /datasets/sft.parquet",
        "--tool-key tools",
        "--metadata-key meta",
        "--load /checkpoints/260827-12345678",
        "--save /checkpoints/260827-12345678",
        "--loss-mask-type qwen3",
        "--tensor-model-parallel-size 8",
        "--expert-model-parallel-size 8",
        "--context-parallel-size 1",
        "--max-tokens-per-gpu 262144",
        "--enable-mtp-training",
        "--moe-token-dispatcher-type flex",
        "--observe-training-entropy",
        "--use-rollout-entropy",
        "--use-prometheus",
        "--prometheus-run-name 260827-12345678",
        "--use-miles-dashboard",
        "--dashboard-forward-prometheus",
        "--dump-details /scratch/260827-12345678/traces",
        "--wandb-project traffic-sft-smoke",
    )
    for fragment in expected_fragments:
        assert fragment in train_command


def test_qwen36_rejects_context_parallelism(monkeypatch, tmp_path: Path) -> None:
    freeze_environment(monkeypatch)
    install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_sft.py")

    with pytest.raises(
        ValueError,
        match="gated-delta layers currently require context_parallel_size=1",
    ):
        call_entrypoint(
            module,
            "execute",
            {
                "model_name": "Qwen3.6-35B-A3B",
                "context_parallel_size": 2,
            },
            sandbox=tmp_path,
        )
