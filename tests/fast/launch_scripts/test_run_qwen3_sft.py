from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


def test_qwen36_two_epoch_long_context_sft(monkeypatch, tmp_path) -> None:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_sft.py")
    call_entrypoint(
        module,
        "execute",
        {
            "model_name": "Qwen3.6-35B-A3B",
            "prompt_data": "/data/mixed.jsonl",
            "num_epoch": 2,
            "global_batch_size": 4,
            "rollout_batch_size": 4,
            "learning_rate": 1e-5,
            "min_learning_rate": 1e-5,
            "save_interval": 560,
            "checkpointed_output_projection": True,
        },
        sandbox=tmp_path,
    )
    command = recording.commands[-1]
    for fragment in (
        "--num-epoch 2",
        "--global-batch-size 4",
        "--rollout-batch-size 4",
        "--save-interval 560",
        "--lr 1e-05",
        "--min-lr 1e-05",
        "--prompt-data /data/mixed.jsonl",
        "--loss-mask-type qwen3",
        "--tensor-model-parallel-size 2",
        "--context-parallel-size 4",
        "--expert-model-parallel-size 8",
        "--sft-checkpointed-output-projection",
        "--log-probs-chunk-size 256",
        "--accumulate-allreduce-grads-in-fp32",
        "--optimizer-cpu-offload",
    ):
        assert fragment in command
    assert "--grad-reduce-in-bf16" not in command
