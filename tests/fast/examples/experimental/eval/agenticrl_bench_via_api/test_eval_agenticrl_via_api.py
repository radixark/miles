import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.experimental.eval.agenticrl_bench_via_api import eval_agenticrl_via_api as eval
from examples.experimental.eval.agenticrl_bench_via_api import sitecustomize


def _args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    monkeypatch.setenv("POLICY_BASE_URL", "https://api.example.com/v1")
    return eval._parse_args(
        [
            "--tasks-dir",
            "/tmp/tasks",
            "--base-url",
            "https://api.example.com/v1",
            *extra,
        ]
    )


def test_select_tasks_keeps_explicit_order(monkeypatch, tmp_path):
    for name in ("b-task", "a-task", "not-a-task"):
        (tmp_path / name).mkdir()
    (tmp_path / "a-task" / "task.toml").write_text("")
    (tmp_path / "b-task" / "task.toml").write_text("")

    tasks = eval._select_tasks(tmp_path, ["b-task", "a-task"])

    assert [path.name for path in tasks] == ["b-task", "a-task"]


def test_select_tasks_rejects_missing_names(tmp_path):
    with pytest.raises(ValueError, match="missing-task"):
        eval._select_tasks(tmp_path, ["missing-task"])


def test_build_command_uses_harbor_and_never_contains_a_key(monkeypatch, tmp_path):
    args = _args(monkeypatch, "--environment-kwarg", "allocate_gpus=true")

    command = eval._build_command(args, Path("/tasks/example"), 2, tmp_path)

    assert command[:2] == ["harbor", "run"]
    assert "kimi_code_agent:KimiCodeAgent" in command
    assert "allocate_gpus=true" in command
    assert command[command.index("--allow-agent-host") + 1] == "api.example.com"
    assert "OPENAI_API_KEY=${OPENAI_API_KEY}" in command
    assert all("secret" not in item for item in command)


def test_child_env_passes_model_configuration_without_logging_secrets(monkeypatch):
    args = _args(monkeypatch, "--system-dns")

    env = eval._child_env(args, "local-secret")

    assert env["OPENAI_API_KEY"] == "local-secret"
    assert env["KIMI_MODEL_BASE_URL"] == "https://api.example.com/v1"
    assert env["KIMI_EVAL_SYSTEM_DNS"] == "1"
    assert env["KIMI_EVAL_STREAM_TRANSFERS"] == "1"
    assert env["PYTHONPATH"].startswith(str(eval.EXAMPLE_DIR))


def test_child_env_loads_the_e2b_key_from_its_file(monkeypatch, tmp_path):
    args = _args(monkeypatch)
    key_file = tmp_path / "e2b-key"
    key_file.write_text("e2b-secret\n")
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    env = eval._child_env(args, "local-secret")

    assert env["E2B_API_KEY"] == "e2b-secret"


def test_environment_kwargs_parse_json_and_reject_invalid_input(monkeypatch):
    args = _args(
        monkeypatch,
        "--environment-kwarg",
        "allocate_gpus=true",
        "--environment-kwarg",
        'nested={"key":1}',
    )

    assert eval._environment_kwargs(args)["allocate_gpus"] is True
    assert eval._environment_kwargs(args)["nested"] == {"key": 1}
    args.environment_kwarg.append("not-a-pair")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        eval._environment_kwargs(args)


def test_read_trial_result_skips_the_job_summary(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"stats": {"n_completed_trials": 1}}))
    trial = tmp_path / "task-attempt"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
            }
        )
    )

    result = eval._read_trial_result(tmp_path)

    assert result == {"reward": 1.0, "exit_status": "Submitted", "trial_dir": str(trial)}


def test_read_trial_result_uses_the_verifier_reward(tmp_path):
    trial = tmp_path / "task-attempt"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": None,
            }
        )
    )

    result = eval._read_trial_result(tmp_path)

    assert result == {"reward": 0.0, "exit_status": "Submitted", "trial_dir": str(trial)}


def test_read_trial_result_reports_an_exception(tmp_path):
    trial = tmp_path / "task-attempt"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": None,
                "exception_info": {"exception_type": "NetworkConnectionError"},
            }
        )
    )

    result = eval._read_trial_result(tmp_path)

    assert result["reward"] is None
    assert result["exit_status"] == "NetworkConnectionError"


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _FakeFiles:
    def __init__(self, chunks):
        self._chunks = chunks
        self.read_kwargs = None
        self.writes = []

    async def read(self, path, **kwargs):
        self.read_kwargs = {"path": path, **kwargs}
        return _FakeStream(self._chunks)

    async def write(self, path, source, **kwargs):
        self.writes.append({"path": path, "data": source.read(), **kwargs})


def test_stream_transfers_use_file_streams_and_gzip(tmp_path):
    files = _FakeFiles([b"streamed ", b"payload"])
    environment = SimpleNamespace(_sandbox=SimpleNamespace(files=files))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "checkpoint.bin").write_bytes(b"checkpoint")

    async def run():
        await sitecustomize._stream_upload_dir(environment, source_dir, "/artifacts")
        await sitecustomize._stream_upload_file(environment, source_dir / "nested" / "checkpoint.bin", "/single.bin")
        await sitecustomize._stream_download_file(environment, "/remote.bin", tmp_path / "downloaded.bin")

    asyncio.run(run())

    assert (tmp_path / "downloaded.bin").read_bytes() == b"streamed payload"
    assert files.read_kwargs == {"path": "/remote.bin", "format": "stream", "gzip": True}
    assert files.writes == [
        {"path": "/artifacts/nested/checkpoint.bin", "data": b"checkpoint", "gzip": True},
        {"path": "/single.bin", "data": b"checkpoint", "gzip": True},
    ]
