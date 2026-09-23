"""Offline checks of the sandbox-smoke harness: the registry's shape and the
axis wiring. The real run needs a sandbox credential and is invoked manually
(see scripts/sandbox_smoke/README.md)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_PY = REPO_ROOT / "scripts" / "sandbox_smoke" / "run.py"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sandbox_smoke_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connectors_are_callables(smoke):
    assert "harbor" in smoke.CONNECTORS
    for name, runner in smoke.CONNECTORS.items():
        assert callable(runner), name


def test_golden_run_wires_all_axes_through(smoke, monkeypatch, tmp_path):
    """connector/backend/agent/task all reach the runner; the golden default needs no endpoint."""
    seen = {}

    async def fake_runner(tasks_dir, task, *, backend, agent, base_url):
        seen.update(tasks_dir=tasks_dir, task=task, backend=backend, agent=agent, base_url=base_url)
        return {"reward": 1.0, "exit_status": "Submitted"}

    monkeypatch.setitem(smoke.CONNECTORS, "harbor", fake_runner)
    monkeypatch.setenv("TB2_TASKS_DIR", str(tmp_path))  # the env var wins over the cached clone
    monkeypatch.setattr("sys.argv", ["run.py", "--connector", "harbor", "--backend", "daytona"])
    assert smoke.main() == 0
    # the benchmark preset filled in the task; the golden default reaches the connector untranslated
    assert seen["backend"] == "daytona" and seen["task"] == "fix-git" and seen["agent"] == smoke.GOLDEN
    assert seen["tasks_dir"] == tmp_path


def test_a_real_harness_requires_a_model_endpoint(smoke, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--connector", "harbor", "--backend", "e2b", "--agent", "mini-swe-agent"]
    )
    with pytest.raises(SystemExit) as excinfo:
        smoke.main()
    assert excinfo.value.code == 2
    assert "--base-url" in capsys.readouterr().err
