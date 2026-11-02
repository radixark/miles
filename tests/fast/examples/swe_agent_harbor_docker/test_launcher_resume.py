"""swe-agent-harbor launchers must resume from their own checkpoints after a restart.

Without --load, miles initializes from --ref-load every time, so a job restarted after a
wall-clock limit or crash retrains from step 0. miles falls back to --ref-load when the
--load dir has no checkpoint yet, so passing --load unconditionally is safe for fresh runs.
"""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = REPO_ROOT / "examples" / "swe-agent-harbor-docker"
# Per-launcher overrides that let execute() run off-cluster.
LAUNCHERS = {
    "run.py": {},
    "run-glm47-flash-agentic-async.py": {"num_nodes": 2},  # needs a node left over for inference
}


def _load_launcher(name: str) -> ModuleType:
    module_name = "swe_harbor_launcher_" + re.sub(r"\W", "_", name)
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train_args(module: ModuleType, monkeypatch: pytest.MonkeyPatch, **overrides) -> str:
    captured = {}
    monkeypatch.setattr(BaseCommandBackend, "execute_train", lambda _self, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")
    module.execute(module.ScriptArgs(**overrides))
    return captured["train_args"]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_resumes_from_save_dir(launcher: str, monkeypatch: pytest.MonkeyPatch) -> None:
    train_args = _train_args(_load_launcher(launcher), monkeypatch, save_dir="/ckpt/run-a/", **LAUNCHERS[launcher])

    assert re.findall(r"--load (\S+)", train_args) == ["/ckpt/run-a/"]
    assert re.findall(r"--save (\S+)", train_args) == ["/ckpt/run-a/"]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_load_dir_overrides_save_dir(launcher: str, monkeypatch: pytest.MonkeyPatch) -> None:
    train_args = _train_args(
        _load_launcher(launcher), monkeypatch, save_dir="/ckpt/run-b/", load_dir="/ckpt/run-a/", **LAUNCHERS[launcher]
    )

    assert re.findall(r"--load (\S+)", train_args) == ["/ckpt/run-a/"]
