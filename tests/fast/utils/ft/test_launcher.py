from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from miles.utils.ft.launcher import _apply_env_vars, main


class TestApplyEnvVars:
    def test_sets_env_vars(self) -> None:
        env = {"FOO": "bar", "BAZ": "123"}
        runtime_json = json.dumps({"env_vars": env})
        with patch.dict("os.environ", {}, clear=True):
            _apply_env_vars(runtime_json)
            import os
            assert os.environ["FOO"] == "bar"
            assert os.environ["BAZ"] == "123"

    def test_no_env_vars_key(self) -> None:
        runtime_json = json.dumps({"working_dir": "/tmp"})
        with patch.dict("os.environ", {}, clear=True):
            _apply_env_vars(runtime_json)

    def test_empty_env_vars(self) -> None:
        runtime_json = json.dumps({"env_vars": {}})
        with patch.dict("os.environ", {}, clear=True):
            _apply_env_vars(runtime_json)


class TestMain:
    def test_execvp_called_with_command(self) -> None:
        argv = [
            "--runtime-env-json", '{"env_vars": {"K": "V"}}',
            "--", "python3", "train.py", "--lr", "0.001",
        ]
        with patch("miles.utils.ft.launcher.os.execvp") as mock_exec:
            main(argv)
        mock_exec.assert_called_once_with(
            "python3", ["python3", "train.py", "--lr", "0.001"],
        )

    def test_env_vars_set_before_exec(self) -> None:
        env = {"MY_VAR": "hello"}
        argv = [
            "--runtime-env-json", json.dumps({"env_vars": env}),
            "--", "echo", "hi",
        ]
        with patch("miles.utils.ft.launcher.os.execvp") as mock_exec, \
             patch.dict("os.environ", {}, clear=True):
            main(argv)
            import os
            assert os.environ["MY_VAR"] == "hello"
        mock_exec.assert_called_once()

    def test_runtime_env_json_equals_form(self) -> None:
        argv = [
            "--runtime-env-json={}", "--", "echo",
        ]
        with patch("miles.utils.ft.launcher.os.execvp") as mock_exec:
            main(argv)
        mock_exec.assert_called_once_with("echo", ["echo"])

    def test_no_command_raises(self) -> None:
        argv = ["--runtime-env-json", "{}"]
        with pytest.raises(SystemExit, match="no command"):
            main(argv)

    def test_unexpected_arg_raises(self) -> None:
        argv = ["--unknown-flag", "--", "echo"]
        with pytest.raises(SystemExit, match="unexpected argument"):
            main(argv)

    def test_no_runtime_env_json_still_execs(self) -> None:
        argv = ["--", "python3", "train.py"]
        with patch("miles.utils.ft.launcher.os.execvp") as mock_exec:
            main(argv)
        mock_exec.assert_called_once_with(
            "python3", ["python3", "train.py"],
        )
