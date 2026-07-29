import json
import os
import shlex

import miles.utils.external_utils.command_utils as command_utils


def test_convert_checkpoint_preserves_source_paths(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
    monkeypatch.setattr(command_utils, "exec_command", commands.append)

    command_utils.convert_checkpoint(
        model_name="model",
        megatron_model_type="model_type",
        num_gpus_per_node=1,
        dir_dst=str(tmp_path),
        megatron_path="/megatron",
    )

    expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/sglang", "/existing"])
    assert f"PYTHONPATH={shlex.quote(expected)} " in commands[0]


def test_execute_train_preserves_source_paths_in_ray_runtime(monkeypatch):
    commands = []
    monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setattr(command_utils, "exec_command", commands.append)
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)

    command_utils.execute_train(
        train_args="",
        num_gpus_per_node=1,
        megatron_model_type="model_type",
        megatron_path="/megatron",
        extra_env_vars={"PYTHONPATH": "/custom:/sglang", "QUOTED_VALUE": "it's preserved"},
    )

    submit_command = commands[-1]
    runtime_env_arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
    runtime_env = json.loads(runtime_env_arg.split("=", 1)[1])
    expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/custom", "/sglang", "/existing"])
    assert runtime_env["env_vars"]["PYTHONPATH"] == expected
    assert runtime_env["env_vars"]["QUOTED_VALUE"] == "it's preserved"
