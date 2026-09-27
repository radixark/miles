"""scripts/amd/run_deepseek_v4.py: what the launcher emits for the GPU it finds.

The one hardware seam is `_probe_device`, which normally runs in a child process; every test pins
it, so none of this needs a GPU.
"""

import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

MI300X = {
    "arch": "gfx942",
    "name": "AMD Instinct MI300X",
    "total_gib": 191.98,
    "hip": True,
    "te_version": "2.8",
    "blockwise": [False, "Float8BlockScaling requires gfx950"],
    "tensorwise": [True, ""],
}
MI355X = {
    "arch": "gfx950",
    "name": "AMD Instinct MI355X",
    "total_gib": 288.0,
    "hip": True,
    "te_version": "2.8",
    "blockwise": [True, ""],
    "tensorwise": [True, ""],
}


@pytest.fixture
def launcher(monkeypatch):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_deepseek_v4.py")
    module.recording = recording
    return module


def _train_argv(launcher, monkeypatch, tmp_path, device, **overrides):
    monkeypatch.setattr(launcher, "_probe_device", lambda: device)
    call_entrypoint(launcher, "train", overrides, sandbox=tmp_path)
    command = launcher.recording.commands[-1]
    assert "ray job submit" in command
    return command


def _flag(command: str, name: str) -> str | None:
    tokens = shlex.split(command.split(" -- python3 ", 1)[1])
    values = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == name]
    assert len(values) <= 1, f"{name} emitted {len(values)} times"
    return values[0] if values else None


def _has(command: str, name: str) -> bool:
    return name in shlex.split(command.split(" -- python3 ", 1)[1])


TWO_MI300X_NODES = dict(num_nodes=2, num_gpus_per_node=8, mode="normal")


def test_two_mi300x_nodes_get_the_verified_layout(launcher, monkeypatch, tmp_path):
    command = _train_argv(launcher, monkeypatch, tmp_path, MI300X, **TWO_MI300X_NODES)

    assert _flag(command, "--tensor-model-parallel-size") == "1"
    assert not _has(command, "--sequence-parallel")
    assert _flag(command, "--pipeline-model-parallel-size") == "2"
    assert _flag(command, "--decoder-first-pipeline-num-layers") == "22"
    assert _flag(command, "--decoder-last-pipeline-num-layers") == "21"
    assert _flag(command, "--context-parallel-size") == "8"
    assert _has(command, "--allgather-cp")
    assert _flag(command, "--distributed-timeout-minutes") == "60"
    assert _flag(command, "--expert-model-parallel-size") == "8"


def test_two_mi300x_nodes_offload_everything_and_stream_the_gradients(launcher, monkeypatch, tmp_path):
    command = _train_argv(launcher, monkeypatch, tmp_path, MI300X, **TWO_MI300X_NODES)

    assert _flag(command, "--optimizer-offload-fraction") == "1.0"
    for flag in (
        "--offload-train",
        "--optimizer-cpu-streaming-gradients",
        "--no-pin-cpu-grads",
        "--no-pin-cpu-params",
        "--rematerialize-param-from-master-weight",
        "--check-rematerialize-param-from-master-weight",
        "--colocate",
    ):
        assert _has(command, flag), flag
    assert _flag(command, "--sglang-mem-fraction-static") == "0.75"
    assert _flag(command, "--train-memory-margin-bytes") == "1073741824"


def test_gfx942_trains_with_the_numerics_its_rollout_uses(launcher, monkeypatch, tmp_path):
    command = _train_argv(launcher, monkeypatch, tmp_path, MI300X, **TWO_MI300X_NODES)

    assert _flag(command, "--fp8-recipe") == "tensorwise"
    assert "NVTE_FP8_BLOCK_SCALING_FP32_SCALES" not in command
    assert _flag(command, "--moe-router-dtype") == "fp32"
    assert _flag(command, "--dsv4-kv-qat") == "off"
    assert _flag(command, "--dsv4-index-qat") == "fp8_dynamic"
    assert _flag(command, "--sglang-moe-runner-backend") == "triton"
    assert _has(command, "--sglang-disable-custom-all-reduce")
    assert _flag(command, "--sglang-router-policy") == "round_robin"
    for env in ('"GPU_MAX_HW_QUEUES": "1"', '"SGLANG_SET_CPU_AFFINITY": "0"', '"TORCHINDUCTOR_MAX_AUTOTUNE": "0"'):
        assert env in command, env


def test_the_dapo_task_is_graded_by_the_answer_format_it_asks_for(launcher, monkeypatch, tmp_path):
    command = _train_argv(launcher, monkeypatch, tmp_path, MI300X, **TWO_MI300X_NODES)

    assert _flag(command, "--rm-type") == "dapo"
    assert _flag(command, "--reward-key") == "score"
    assert _flag(command, "--eval-reward-key") == "acc"
    assert _flag(command, "--sglang-context-length") == "16384"
    assert int(_flag(command, "--rollout-max-response-len")) < 16384


def test_aime_eval_is_graded_on_the_boxed_answer(launcher, monkeypatch, tmp_path):
    """AIME prompts do not ask for "Answer:", so the DAPO grader would score every sample -1."""
    command = _train_argv(launcher, monkeypatch, tmp_path, MI300X, **TWO_MI300X_NODES)

    assert _flag(command, "--eval-prompt-data") is None
    assert _flag(command, "--eval-config").startswith("base64:")
    (config,) = [text for text in launcher.recording.pseudo_files if "aime-2024.jsonl" in text]
    assert "rm_type: dapo_boxed" in config


def test_gfx950_keeps_its_verified_four_node_recipe(launcher, monkeypatch, tmp_path):
    command = _train_argv(launcher, monkeypatch, tmp_path, MI355X, num_nodes=4, num_gpus_per_node=8, mode="normal")

    assert _flag(command, "--fp8-recipe") == "blockwise"
    assert "NVTE_FP8_BLOCK_SCALING_FP32_SCALES" in command
    assert _flag(command, "--tensor-model-parallel-size") == "4"
    assert _flag(command, "--pipeline-model-parallel-size") == "4"
    assert _flag(command, "--context-parallel-size") == "1"
    assert _flag(command, "--distributed-timeout-minutes") is None
    assert _flag(command, "--optimizer-offload-fraction") == "0.75"
    assert _has(command, "--no-offload-train")
    assert _flag(command, "--sglang-mem-fraction-static") == "0.5"
    assert _flag(command, "--train-memory-margin-bytes") == "3221225472"
    for gfx942_only in (
        "--moe-router-dtype",
        "--dsv4-kv-qat",
        "--dsv4-index-qat",
        "--sglang-moe-runner-backend",
        "--sglang-router-policy",
        "--optimizer-cpu-streaming-gradients",
    ):
        assert not _has(command, gfx942_only), gfx942_only
    assert "GPU_MAX_HW_QUEUES" not in command


def test_two_actor_nodes_without_colocate_do_not_get_the_colocate_only_rebuild(launcher, monkeypatch, tmp_path):
    """--rematerialize-param-from-master-weight asserts --colocate; a 2+1 split must not emit it."""
    command = _train_argv(
        launcher, monkeypatch, tmp_path, MI300X, num_nodes=3, rollout_num_nodes=1, num_gpus_per_node=8, mode="normal"
    )

    assert _flag(command, "--actor-num-nodes") == "2"
    assert not _has(command, "--colocate")
    assert not _has(command, "--rematerialize-param-from-master-weight")


def test_the_mem_fraction_override_wins_and_is_passed_through_exactly(launcher, monkeypatch, tmp_path):
    command = _train_argv(
        launcher, monkeypatch, tmp_path, MI300X, sglang_mem_fraction_static=0.7125, **TWO_MI300X_NODES
    )

    assert _flag(command, "--sglang-mem-fraction-static") == "0.7125"


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.1])
def test_an_out_of_range_mem_fraction_is_rejected_before_anything_runs(launcher, monkeypatch, value):
    monkeypatch.setattr(launcher, "_probe_device", lambda: pytest.fail("probed the GPU before validating"))
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        launcher.ScriptArgs(sglang_mem_fraction_static=value, **TWO_MI300X_NODES)
    assert launcher.recording.commands == []


def test_an_unsupported_explicit_recipe_fails_before_any_command(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_probe_device", lambda: MI300X)
    with pytest.raises(RuntimeError, match="no automatic BF16 fallback"):
        call_entrypoint(launcher, "full_train", {"fp8_recipe": "blockwise", **TWO_MI300X_NODES}, sandbox=tmp_path)
    assert launcher.recording.commands == []


def test_bf16_training_does_not_probe_te(launcher, monkeypatch, tmp_path):
    no_te = dict(MI300X, te_version=None, blockwise=[False, "no TE"], tensorwise=[False, "no TE"])
    command = _train_argv(launcher, monkeypatch, tmp_path, no_te, fp8_training=False, **TWO_MI300X_NODES)

    assert _flag(command, "--fp8-recipe") is None
    assert not _has(command, "--fp8-format")


@pytest.mark.parametrize("cp_size", [3, 5])
def test_a_context_parallel_size_that_does_not_divide_the_gpus_is_refused(launcher, monkeypatch, tmp_path, cp_size):
    monkeypatch.setattr(launcher, "_probe_device", lambda: MI300X)
    with pytest.raises(NotImplementedError, match="does not divide"):
        call_entrypoint(launcher, "train", {"context_parallel_size": cp_size, **TWO_MI300X_NODES}, sandbox=tmp_path)


def test_context_parallelism_is_refused_on_one_node(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_probe_device", lambda: MI300X)
    with pytest.raises(NotImplementedError, match="multi-node"):
        call_entrypoint(
            launcher,
            "train",
            {"num_nodes": 1, "num_gpus_per_node": 8, "context_parallel_size": 8, "mode": "normal"},
            sandbox=tmp_path,
        )


def test_train_without_hf_checkpoint_reads_model_local_dir(launcher, monkeypatch, tmp_path):
    command = _train_argv(
        launcher,
        monkeypatch,
        tmp_path,
        MI300X,
        model_dir="/nfs/models",
        model_local_dir="/nvme/models",
        **TWO_MI300X_NODES,
    )

    assert _flag(command, "--hf-checkpoint") == "/nvme/models/DeepSeek-V4-Flash-FP8"
