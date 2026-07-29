import json
import os
import shlex

import pytest

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


def test_execute_train_runs_hook_after_ray_restart_and_before_submit(monkeypatch):
    events = []
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "0")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setattr(command_utils, "exec_command", lambda command: events.append(("command", command)))
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)

    command_utils.execute_train(
        train_args="",
        num_gpus_per_node=1,
        megatron_model_type="model_type",
        before_ray_job_submit=lambda: events.append(("hook", None)),
    )

    assert [event[0] for event in events] == ["command", "command", "hook", "command"]
    assert "pkill -9 sglang" in events[0][1]
    assert "ray start --head" in events[1][1]
    assert "ray job submit" in events[3][1]


def test_start_mooncake_master_reuses_ready_server(monkeypatch):
    commands = []
    waits = []
    monkeypatch.setattr(command_utils, "_is_tcp_server_ready", lambda host, port: True)
    monkeypatch.setattr(command_utils, "exec_command", commands.append)
    monkeypatch.setattr(command_utils, "wait_for_server_ready", lambda *args, **kwargs: waits.append((args, kwargs)))

    command_utils.start_mooncake_master()

    assert commands == []
    assert waits == []


def test_start_mooncake_master_restarts_and_waits_until_ready(monkeypatch, tmp_path):
    commands = []
    waits = []
    log_path = tmp_path / "mooncake master.log"
    monkeypatch.setattr(command_utils, "_is_tcp_server_ready", lambda host, port: False)
    monkeypatch.setattr(command_utils, "exec_command", commands.append)
    monkeypatch.setattr(command_utils, "wait_for_server_ready", lambda *args, **kwargs: waits.append((args, kwargs)))

    command_utils.start_mooncake_master(rpc_port=50151, metrics_port=50152, timeout=12, log_path=log_path)

    assert len(commands) == 1
    assert "pkill -x mooncake_master" in commands[0]
    assert "mooncake_master --rpc_port 50151 --metrics_port 50152" in commands[0]
    assert f"> {shlex.quote(str(log_path))} 2>&1 &" in commands[0]
    assert waits == [(("127.0.0.1", 50151), {"timeout": 12})]


def test_start_mooncake_master_reports_log_when_startup_fails(monkeypatch, tmp_path):
    log_path = tmp_path / "mooncake_master.log"
    log_path.write_text("bind failed\nfatal startup error\n")
    commands = []
    monkeypatch.setattr(command_utils, "_is_tcp_server_ready", lambda host, port: False)
    monkeypatch.setattr(command_utils, "exec_command", commands.append)

    def fail_wait(*args, **kwargs):
        raise RuntimeError("not ready")

    monkeypatch.setattr(command_utils, "wait_for_server_ready", fail_wait)

    with pytest.raises(RuntimeError, match="fatal startup error"):
        command_utils.start_mooncake_master(log_path=log_path)

    assert len(commands) == 2
    assert all("pkill -x mooncake_master" in command for command in commands)
