"""Offline evaluation contracts; the GPU and sandbox are not started."""

import asyncio
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from examples.experimental.tmax_score_centering.prepare_eval_data import prepare


def test_eval_preparation_preserves_tasks_without_leaking_verifiers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "instruction.md").write_text("Create /workspace/result.txt.")
    config = '[agent]\ntimeout_sec = 3600\n[environment]\ncpus = 4\nmemory = "8G"\n'
    (source / "task.toml").write_text(config)
    (source / "environment").mkdir()
    (source / "environment" / "Dockerfile").write_text("FROM example\nWORKDIR /workspace\n")
    (source / "tests").mkdir()
    (source / "tests" / "test.sh").write_text("SECRET_VERIFIER_CONTENT")
    (source / "solution").mkdir()
    (source / "solution" / "solve.sh").write_text("SECRET_ORACLE_CONTENT")
    spec = {
        "name": "terminal-bench",
        "version": "2.0",
        "tasks": [{"name": "test-task", "git_url": "https://example/repo", "git_commit_id": "abc"}],
    }
    output = tmp_path / "eval.jsonl"
    tasks = tmp_path / "tasks"
    assert prepare(spec, {"test-task": source}, output, tasks) == 1
    row = json.loads(output.read_text())
    assert row["metadata"]["split"] == "eval"
    assert row["metadata"]["git_commit_id"] == "abc"
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert "Create /workspace/result.txt." in row["prompt"][1]["content"]
    assert "SECRET_" not in output.read_text()
    copied = tasks / row["metadata"]["instance_id"]
    assert (copied / "task.toml").read_text() == config
    assert (copied / "tests" / "test.sh").read_bytes() == (source / "tests" / "test.sh").read_bytes()
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == spec


@pytest.fixture
def agent_module(monkeypatch):
    """Mock the optional Harbor package, retaining the actual Miles adapter."""

    class Config(SimpleNamespace):
        def model_copy(self, *, update):
            return Config(**{**vars(self), **update})

    modules = {
        "harbor.agents.base": {"BaseAgent": Config},
        "harbor.environments.base": {"BaseEnvironment": Config},
        "harbor.models.agent.context": {"AgentContext": Config},
        "harbor.models.trial.config": {"AgentConfig": Config, "VerifierConfig": Config},
        "harbor.trial.trial": {"Trial": Config},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    name = "examples.experimental.tmax_score_centering.agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module, Config
    sys.modules.pop(name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation,expected_reward", [(True, 1.0), (False, 0.0)])
async def test_eval_uses_verifier_and_task_budgets_without_changing_training(
    agent_module, monkeypatch, tmp_path, evaluation, expected_reward
):
    agent, Config = agent_module
    original_env = Config(override_cpus=1, override_memory_mb=2048, override_storage_mb=10240)
    config = Config(environment=original_env, verifier=Config(override_timeout_sec=600), timeout_multiplier=2)
    monkeypatch.setattr(agent, "build_trial_config", lambda *args: config)
    monkeypatch.setattr(agent, "_ensure_provider_key", lambda: None)
    monkeypatch.setenv("TMAX_MAX_STEPS", "16")
    monkeypatch.setenv("TMAX_EVAL_MAX_STEPS", "64")
    monkeypatch.setenv("AGENT_TIMEOUT", "900")
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0.001")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "tmax.json").write_text(json.dumps({"submitted": False, "turns": 16}))
    result = Config(exception_info=None, verifier_result=Config(rewards={"reward": 1.0}))

    class Trial:
        paths = Config(trial_dir=tmp_path)

        @classmethod
        async def create(cls, received):
            assert received is config
            return cls()

        async def run(self):
            return result

    monkeypatch.setattr(agent, "Trial", Trial)
    verdict = await agent.run(
        "http://session",
        [{"role": "system"}, {"role": "user"}],
        {},
        {"split": "eval" if evaluation else "train"},
    )
    assert verdict["reward"] == expected_reward
    assert config.agent.kwargs["max_steps"] == (64 if evaluation else 16)
    assert config.agent.kwargs["preserve_workdir"] is evaluation
    assert original_env.override_cpus == 1
    assert config.environment.import_path.endswith(":TMaxE2BEnvironment")
    if evaluation:
        assert config.agent.override_timeout_sec is None
        assert config.environment.override_cpus is None
        assert config.environment.override_memory_mb is None
        assert config.environment.override_storage_mb is None
        assert not vars(config.verifier)
        assert config.timeout_multiplier == 1
    else:
        assert config.agent.override_timeout_sec == 900
        assert config.environment.override_cpus == 1
        assert verdict["exit_status"] == "StepOrTokenLimitExceeded"


@pytest.mark.asyncio
async def test_eval_shell_starts_in_task_workdir(agent_module, monkeypatch, tmp_path):
    agent, Config = agent_module
    state = tmp_path / "state"
    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    monkeypatch.setattr(agent, "_STATE", str(state))

    class Environment:
        async def exec(self, *, command, timeout_sec):
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=workdir,
                timeout=timeout_sec,
                env={"PATH": "/usr/bin:/bin", "TASK_MARKER": "preserved"},
                capture_output=True,
                text=True,
            )
            return Config(return_code=result.returncode, stderr=result.stderr)

    instance = agent.TMaxAgent(
        logs_dir=tmp_path,
        model_name="model",
        messages=[],
        api_base="http://unused",
        request_kwargs={},
        preserve_workdir=True,
    )
    await instance.setup(Environment())
    assert Path((state / "cwd").read_text().strip()).resolve() == workdir.resolve()
    assert 'TASK_MARKER="preserved"' in (state / "env").read_text()


@pytest.mark.asyncio
async def test_sibling_episodes_build_one_template(monkeypatch):
    class Environment:
        builds = 0
        sandboxes = 0
        ready = False
        _template_name = "cold-test-template"
        _prebuilt_template_id = None

        async def _does_template_exist(self):
            return Environment.ready

        async def _create_template(self):
            Environment.builds += 1
            await asyncio.sleep(0.01)
            Environment.ready = True

        async def start(self, force_build):
            if force_build or not await self._does_template_exist():
                await self._create_template()
            Environment.sandboxes += 1

    stub = ModuleType("harbor.environments.e2b")
    stub.E2BEnvironment = Environment
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    name = "examples.experimental.tmax_score_centering.environment"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    try:
        await asyncio.gather(*(module.TMaxE2BEnvironment().start(False) for _ in range(8)))
        assert Environment.builds == 1
        assert Environment.sandboxes == 8
    finally:
        sys.modules.pop(name, None)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("timeout") is None, reason="requires Linux coreutils, as the task images do")
async def test_command_timeout_returns_partial_output_and_stops_children(monkeypatch, tmp_path):
    class Environment:
        async def exec(self, command, *, timeout_sec, **kwargs):
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            return SimpleNamespace(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)

    stub = ModuleType("harbor.environments.e2b")
    stub.E2BEnvironment = Environment
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    name = "examples.experimental.tmax_score_centering.environment"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    try:
        marker = tmp_path / "late-side-effect"
        instance = module.TMaxE2BEnvironment()
        result = await instance.exec(f"printf partial-output; (sleep 0.2; touch {marker}) & wait", timeout_sec=0.05)
        assert result.return_code == 124
        assert result.stdout == "partial-output"
        await asyncio.sleep(0.3)
        assert not marker.exists()
        result = await instance.exec("printf still-alive", timeout_sec=1)
        assert result.return_code == 0
        assert result.stdout == "still-alive"
        # A background service must survive, but must not hold the tool's
        # output stream open after its foreground shell returns.
        result = await asyncio.wait_for(
            instance.exec(f"(sleep 2; touch {marker}) & printf launched; printf diagnostic >&2", timeout_sec=1),
            timeout=1,
        )
        assert result.return_code == 0
        assert result.stdout == "launched"
        assert result.stderr == "diagnostic"
        await asyncio.sleep(2.2)
        assert marker.exists()
    finally:
        sys.modules.pop(name, None)
