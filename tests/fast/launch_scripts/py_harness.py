import ast
import importlib.util
import inspect
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, sanitize
from tests.fast.utils.command_recorder import record_commands

import miles.utils.external_utils.command_utils as command_utils

FROZEN_RUN_ID = "260101-000000-000"

_FROZEN_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
    "PYTHONPATH": "/frozen/pythonpath",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

_CLEARED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "GITHUB_COMMIT_NAME",
    "GLOO_SOCKET_IFNAME",
    "KEEP_MOE_LORA",
    "MILES_SCRIPT_EXTERNAL_RAY",
    "NCCL_DEBUG",
    "NCCL_DEBUG_FILE",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "NO_PROXY",
    "OPTIMIZER_CPU_OFFLOAD",
    "RAY_ADDRESS",
    "SLURM_JOB_NUM_NODES",
)


@dataclass(frozen=True)
class Recording:
    commands: list[str]
    pseudo_files: list[str]


@dataclass(frozen=True)
class PyLaunchScript:
    path: Path
    entrypoints: tuple[str, ...]

    @property
    def rel(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


def iter_py_launch_scripts() -> list[PyLaunchScript]:
    paths = sorted((REPO_ROOT / "scripts").rglob("run_*.py"))
    return [PyLaunchScript(path=path, entrypoints=tuple(_entrypoint_names(path))) for path in paths]


def freeze_environment(monkeypatch) -> None:
    for key, value in _FROZEN_ENV.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


def install_command_recorder(monkeypatch) -> Recording:
    recording = Recording(commands=record_commands(monkeypatch), pseudo_files=[])

    def fake_encode_pseudo_file(text: str) -> str:
        recording.pseudo_files.append(text)
        return f"base64:<frozen-pseudo-file-{len(recording.pseudo_files)}>"

    monkeypatch.setattr(command_utils, "create_run_id", lambda: FROZEN_RUN_ID)
    monkeypatch.setattr(command_utils, "encode_pseudo_file", fake_encode_pseudo_file)

    return recording


def import_launch_script(path: Path) -> ModuleType:
    name = "miles_launch_script_" + path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


@contextmanager
def host_filesystem_frozen(sandbox: Path) -> Iterator[None]:
    """Launchers skip work whose artifact already exists, so only the checkout and the sandbox may be visible.

    Without this the recording depends on which checkpoints the machine happens to carry, and on
    python 3.11 a `/root` path the user cannot stat raises PermissionError instead of reporting
    absence. The checkout stays visible because a launcher legitimately resolves its own model args
    script out of it.
    """
    visible_roots = (sandbox, REPO_ROOT)
    real_exists = Path.exists

    def exists(self: Path, **kwargs: object) -> bool:
        if any(self == root or self.is_relative_to(root) for root in visible_roots):
            return real_exists(self, **kwargs)
        return False

    Path.exists = exists
    try:
        yield
    finally:
        Path.exists = real_exists


def call_entrypoint(module: ModuleType, name: str, overrides: dict[str, object], sandbox: Path) -> None:
    entrypoint = getattr(module, name)
    first = next(iter(inspect.signature(entrypoint).parameters.values()), None)
    with host_filesystem_frozen(sandbox):
        if first is not None and first.name == "args":
            entrypoint(module.ScriptArgs(**overrides))
        else:
            entrypoint(**overrides)


def format_recording(recording: Recording, sandbox: Path) -> str:
    """The generated config files are the training recipe, so a snapshot that omits them proves little."""
    lines = []
    for index, command in enumerate(recording.commands):
        lines.append(f"### {index}")
        lines.append(re.sub(r" (?=--)", "\n  ", sanitize(command, sandbox=sandbox)))
        lines.append("")
    for index, content in enumerate(recording.pseudo_files, start=1):
        lines.append(f"### pseudo file {index}")
        lines.append(sanitize(content, sandbox=sandbox))
        lines.append("")
    return "\n".join(lines)


def _entrypoint_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name != "main"
    ]
