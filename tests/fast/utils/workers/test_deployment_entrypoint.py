from __future__ import annotations

import ast
import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.workers import deployment_entrypoint


class _FakeDeployment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def launch_worker_manager(self, args) -> str:
        self.calls.append("launch_worker_manager")
        return "worker-manager"

    def compute_specs(self, args) -> list[SimpleNamespace]:
        self.calls.append("compute_specs")
        return [SimpleNamespace(name="trainer-controller")]


def _injected(monkeypatch, fake: _FakeDeployment) -> None:
    monkeypatch.setattr(deployment_entrypoint, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deployment_entrypoint, "launch_worker_manager", fake.launch_worker_manager)
    monkeypatch.setattr(deployment_entrypoint, "compute_specs", fake.compute_specs)


class TestServeDeployedWorkers:
    async def test_it_launches_the_run_s_workers_and_then_stays_up(self, monkeypatch):
        """This deployment has no training to finish, so serving it means launching and then never returning."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        task = asyncio.ensure_future(
            deployment_entrypoint._serve_deployed_workers(Namespace(deploy_component="trainer"))
        )
        for _ in range(100):
            await asyncio.sleep(0)
            if len(fake.calls) == 2:
                break

        assert fake.calls == ["launch_worker_manager", "compute_specs"]
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_deployment_that_carries_the_orchestration_script_is_refused(self, monkeypatch):
        """Such a deployment is started by its driver script, and serving it here would run it twice."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        with pytest.raises(AssertionError, match="carries no orchestration script"):
            await deployment_entrypoint._serve_deployed_workers(Namespace(deploy_component="all"))

        assert fake.calls == []


_DRIVER_SCRIPTS = [
    "train.py",
    "train_async.py",
    "train_multi_lora_async.py",
    "train_multi_policy.py",
]


class TestEveryDriverRunsItsOwnOrchestrationScript:
    @pytest.mark.parametrize("script", _DRIVER_SCRIPTS)
    def test_a_driver_starts_its_training_without_asking_what_this_launch_deploys(self, script):
        """A deployment carrying no orchestration script is served by its own entrypoint, never by a driver."""
        assert "asyncio.run" in _functions_called_in_main(script)
        assert "deploy_component" not in (REPO_ROOT / script).read_text()


def _functions_called_in_main(script: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / script).read_text())
    return {
        ast.unparse(node.func)
        for statement in tree.body
        if isinstance(statement, ast.If)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    }
