"""Every swe-agent-harbor launcher must cap the engine's per-request context.

--max-seq-len only trims collected samples after a trial ends. Without
--sglang-context-length the engine serves up to the model's own window, so one runaway
trajectory can hold a KV footprint several times larger than intended.
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
    "run_glm52_lora_tb2_daytona.py": {"save_dir": "{tmp}/save/"},  # writes an sglang config under save_dir
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
def test_launcher_caps_engine_context(launcher: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_launcher(launcher)
    overrides = {k: v.format(tmp=tmp_path) if isinstance(v, str) else v for k, v in LAUNCHERS[launcher].items()}

    train_args = _train_args(module, monkeypatch, sglang_context_length=12345, **overrides)

    assert re.findall(r"--sglang-context-length (\d+)", train_args) == ["12345"]


@pytest.mark.parametrize("launcher", ["run.py", "run-glm47-flash-agentic-async.py"])
def test_default_context_leaves_room_for_one_turn_past_max_seq_len(
    launcher: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine cap must sit above max_seq_len, or the max_seq_len limit can never be reached."""
    module = _load_launcher(launcher)
    overrides = {k: v.format(tmp=tmp_path) if isinstance(v, str) else v for k, v in LAUNCHERS[launcher].items()}

    train_args = _train_args(module, monkeypatch, max_seq_len=20000, rollout_max_response_len=3000, **overrides)

    assert re.findall(r"--sglang-context-length (\d+)", train_args) == ["23000"]
    assert re.findall(r"--rollout-max-response-len (\d+)", train_args) == ["3000"]
