import ast
import inspect
import os
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, sanitize
from tests.fast.utils.command_recorder import record_commands

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.model_args_utils import _import_module_from_path

FROZEN_RUN_ID = "260101-000000-000"
FROZEN_HARDWARE = "H200"

_GPU_COUNT_ANY_WAIT_LOOP_ACCEPTS = "1000000"
_FROZEN_PID = 1000
_FROZEN_PPID = 1001

_FROZEN_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
    "PYTHONPATH": "/frozen/pythonpath",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

CLEARED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "GITHUB_COMMIT_NAME",
    "GLOO_SOCKET_IFNAME",
    "KEEP_MOE_LORA",
    "MILES_SCRIPT_EXTERNAL_RAY",
    "MILES_USE_LEGACY_ROLLOUT_V1",
    "MLP_SOCKET_IFNAME",
    "MLP_WORKER_0_HOST",
    "MODEL_ARGS_FIRST_K_DENSE_REPLACE",
    "MODEL_ARGS_NUM_LAYERS",
    "MODEL_ARGS_ROTARY_BASE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_FILE",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "NO_PROXY",
    "OPTIMIZER_CPU_OFFLOAD",
    "RAY_ADDRESS",
    "ROTARY_SCALING_FACTOR",
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


def launcher_hardware_literals() -> dict[str, tuple[str, ...]]:
    """Every value each launcher's `--hardware` accepts, for the launchers that have the flag at all."""
    literals = {}
    for script in iter_py_launch_scripts():
        for node in ast.walk(ast.parse(script.path.read_text())):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "hardware":
                values = node.annotation.slice
                literals[script.rel] = tuple(
                    e.value for e in (values.elts if isinstance(values, ast.Tuple) else [values])
                )
    return literals


def iter_self_executing_launchers() -> list[Path]:
    """Launchers that reach the shell themselves rather than through command_utils."""
    roots = [REPO_ROOT / root for root in ("scripts", "examples", "tools")]
    convention = {script.path for script in iter_py_launch_scripts()}
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if path not in convention and "ray job submit" in path.read_text(errors="replace")
    )


def install_shell_recorder(monkeypatch, sandbox: Path) -> Recording:
    """A launcher holding its own subprocess handle never touches the recorded command_utils helpers."""
    recording = Recording(commands=[], pseudo_files=[])

    def fake_run(command, *args, **kwargs):
        recording.commands.append(command if isinstance(command, str) else " ".join(command))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=_GPU_COUNT_ANY_WAIT_LOOP_ACCEPTS, stderr=""
        )

    monkeypatch.setenv("MILES_LOG_DIR", str(sandbox))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(os, "makedirs", lambda path, **kwargs: None)
    monkeypatch.setattr(os, "getpid", lambda: _FROZEN_PID)
    monkeypatch.setattr(os, "getppid", lambda: _FROZEN_PPID)

    return recording


def freeze_environment(monkeypatch, hardware: str = FROZEN_HARDWARE) -> None:
    for key, value in _FROZEN_ENV.items():
        monkeypatch.setenv(key, value)
    for key in CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(base_backend, "detect_hardware", lambda: hardware)


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
    return _import_module_from_path(path, name)


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
    saved_env = dict(os.environ)
    try:
        with host_filesystem_frozen(sandbox):
            if first is not None and first.name == "args":
                entrypoint(module.ScriptArgs(**overrides))
            else:
                entrypoint(**overrides)
    finally:
        # a leaked knob would make later recordings depend on which launcher ran first
        os.environ.clear()
        os.environ.update(saved_env)


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
