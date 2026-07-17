"""Per-task sandboxes for Terminal-Bench-2.

TB2 is a per-task-image benchmark: every task pins its official runtime image
in ``task.toml`` (``[environment].docker_image`` — the exact image the TB2
harness itself runs). A cloud sandbox serving this env must therefore be built
per task: **the official task image ⊕ this env's server layer**, one layer, no
DinD. This module owns that recipe.

Provider-agnostic core:
  ``_server_layer_commands(task_dir)``  shell commands that turn the official
      task image into a combined task+env-server image: a uv-managed Python
      venv at ``/opt/envserver`` running the INSTALLED tbench2_env package
      (source embedded into the build, so a patched checkout ships without a
      released package), plus the task directory staged at ``/opt/tb2-tasks/<id>``
      (downloaded from the tasks checkout's pinned-commit GitHub tarball) for
      ``reset(task_id)`` via ``TB2_TASKS_DIR``. These are plain shell
      commands — nothing Daytona-specific — so the same recipe can back a
      Dockerfile or another provider's build.
  ``server_cmd()``  starts the env server inside the sandbox. Sets
      ``CAMEL_RUNTIME=true`` so camel's TerminalToolkit runs commands in the
      task image's native environment instead of hijacking PATH with its own
      Python 3.10 ``.initial_env`` venv (real TB2 agents see the image's
      python), and ``TB2_COMMAND_TIMEOUT_S`` for realistic command budgets.

Daytona materialization:
  ``create_task_sandbox(...)``  per-episode declarative create straight from
      the ``Image`` definition. Named snapshots count against an org-level
      quota, so registering one per task may not scale to a full task suite;
      the declarative path avoids the quota entirely, and repeat creates hit
      Daytona's build cache (~1min after the first build). Daytona does not
      run the image CMD, so this execs ``server_cmd()`` and waits for /health.
  bake CLI (``python tb2_task_sandbox.py ...``)  optionally pre-register
      named snapshots ``<prefix><task-id>`` as a warm cache.

Verifier-asset hygiene (mirrors the official harness's stage-at-verify
model): ``solution/`` is excluded from the staged task directory at build
time (nothing in this env ever reads it — it exists only for oracle runs),
and ``SERVER_CMD`` sets ``TB2_WITHHOLD_TESTS=1`` so the server pulls
``tests/`` into process memory at ``reset()`` and deletes it from disk before
the agent's first action; a pristine copy is staged at ``/tests`` only for
the verify window. Residual risk, shared with the official TB2 harness:
verification necessarily runs inside the same container the agent controlled
(task state — services, git state — is not portable), so a root agent that
tampers with container binaries could still fake a pass.
"""

import argparse
import base64
import io
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

# Guard for the ONE payload that must be embedded into the build: the
# tbench2_env package source (a patched checkout exists nowhere downloadable).
# The hard ceiling is Daytona's Dockerfile parser: a single line may not
# exceed 65535 bytes ("dockerfile line greater than max allowed size",
# observed on real builds), and base64 inflates by 4/3; 45KB of tar.gz stays
# safely under that. Task directories are never embedded — they download from
# the pinned-SHA GitHub tarball instead (see _task_layer_command).
_MAX_INLINE_TAR_BYTES = 45_000

# What `pip install <dir>` needs from the package checkout; everything else
# (uv.lock, caches) stays out of the image.
_ENV_SRC_ITEMS = (
    "pyproject.toml",
    "README.md",
    "openenv.yaml",
    "__init__.py",
    "client.py",
    "models.py",
    "server",
)

# The command to start the env server inside a task sandbox (Daytona does not
# run the image CMD). /opt/envserver and /opt/tb2-tasks are baked by
# _server_layer_commands.
SERVER_CMD = (
    "TB2_TASKS_DIR=/opt/tb2-tasks "
    "TB2_DEFAULT_TASK_ID={default_task_id} "
    "TB2_COMMAND_TIMEOUT_S={command_timeout_s} "
    "TB2_WITHHOLD_TESTS=1 "
    "CAMEL_RUNTIME=true "
    "MAX_CONCURRENT_ENVS=1 "
    "/opt/envserver/bin/python -m uvicorn tbench2_env.server.app:app "
    "--host 0.0.0.0 --port 8000"
)


def server_cmd(command_timeout_s: int = 900, default_task_id: str = "") -> str:
    # A per-task sandbox stages exactly one task, so make it the default:
    # a reset() with no task_id resolves to the staged task rather than the
    # env's built-in headless-terminal default (which isn't present here).
    return SERVER_CMD.format(
        command_timeout_s=command_timeout_s,
        default_task_id=shlex.quote(default_task_id),
    )


def snapshot_name(prefix: str, task_id: str) -> str:
    return prefix + re.sub(r"[^a-z0-9-]", "-", task_id.lower())


def read_task_config(task_dir: Path) -> dict:
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"{task_dir}: no task.toml")
    return tomllib.loads(toml_path.read_text())


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = Path(tarinfo.name).name
    if name in {"__pycache__", ".initial_env"} or name.endswith((".pyc", ".egg-info")):
        return None
    return tarinfo


def _dir_tar_b64(paths: list[Path], arcnames: list[str], max_bytes: int) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, arcname in zip(paths, arcnames, strict=True):
            tar.add(path, arcname=arcname, filter=_tar_filter)
    raw = buf.getvalue()
    if len(raw) > max_bytes:
        raise ValueError(f"embedded tar is {len(raw)} bytes (> {max_bytes}); " "inline embedding not suitable.")
    return base64.b64encode(raw).decode()


def _task_layer_command(task_dir: Path) -> str:
    """Build command that stages the task dir at /opt/tb2-tasks/<id>.

    One uniform path for every task: download the checkout's pinned-commit
    GitHub tarball and extract just this task. Deterministic (the SHA pins
    the content — note: the committed tree, not uncommitted local edits) and
    payload-free, so build commands stay far from Daytona's 64KB
    Dockerfile-line ceiling regardless of task-dir size. Requires the tasks
    checkout to be a git clone with a GitHub origin.
    """
    repo_root = task_dir.parent
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    m = re.search(r"github\.com[:/]+([^/]+)/([^/.]+)", remote)
    if not m:
        raise ValueError(
            f"{task_dir.name}: the tasks checkout's origin ({remote}) is not "
            "a GitHub remote to download the task tarball from."
        )
    owner, repo = m.group(1), m.group(2)
    url = f"https://github.com/{owner}/{repo}/archive/{sha}.tar.gz"
    # solution/ never enters the image: nothing in this env reads it (it
    # exists only for oracle runs), and an agent must not be able to cat the
    # answer out of the staged task directory.
    prefix = f"{repo}-{sha}/{task_dir.name}"
    return (
        f"mkdir -p /opt/tb2-tasks && curl -fsSL {url} | "
        f"tar xz --strip-components=1 -C /opt/tb2-tasks "
        f"--exclude='{prefix}/solution' --exclude='{prefix}/solution/*' '{prefix}'"
    )


def _env_src_dir() -> Path:
    """Directory of the installed tbench2_env package source.

    The build embeds the package source into the image (_env_src_tar_b64),
    including pyproject.toml so ``pip install <dir>`` works inside the build.
    That requires tbench2_env to be installed editable from a checkout
    (``pip install -e <OpenEnv>/envs/tbench2_env``), where the package
    directory IS the project directory; a wheel/sdist install ships no
    pyproject.toml, so fail fast here instead of deep inside the image build.
    """
    import tbench2_env

    src = Path(tbench2_env.__file__).resolve().parent
    if not (src / "pyproject.toml").is_file():
        raise RuntimeError(
            f"tbench2_env at {src} has no pyproject.toml; the per-task sandbox "
            "build embeds the package source and needs an editable/checkout "
            "install: pip install -e <OpenEnv>/envs/tbench2_env"
        )
    return src


def _env_src_tar_b64() -> str:
    src_dir = _env_src_dir()
    paths, arcnames = [], []
    for item in _ENV_SRC_ITEMS:
        p = src_dir / item
        if p.exists():
            paths.append(p)
            arcnames.append(f"tbench2_env_src/{item}")
    return _dir_tar_b64(paths, arcnames, _MAX_INLINE_TAR_BYTES)


def _resolve_docker_image(task_dir: Path, docker_image: str | None) -> str:
    if docker_image:
        return docker_image
    docker_image = read_task_config(task_dir).get("environment", {}).get("docker_image")
    if not docker_image:
        raise ValueError(f"{task_dir.name}: task.toml has no [environment].docker_image")
    return docker_image


def _server_layer_commands(task_dir: Path) -> list[str]:
    """The provider-agnostic recipe: shell commands that turn the OFFICIAL task
    image into a combined task+env-server image. Nothing here is
    Daytona-specific — the same layers work for docker build, Modal, ACA, etc.
    """
    return [
        # Server-layer OS deps. Task images are heterogeneous; assume
        # debian-ish (all 89 current TB2 images are debian/ubuntu based). Do
        # NOT rm /var/lib/apt/lists afterwards: the official task image's apt
        # state is part of the task environment — solutions and agents run
        # bare `apt install` relying on the index the task image baked in.
        # (curl/ca-certificates/bash stay installed: debian-ish images almost
        # always ship them anyway, and the official test.sh apt-installs curl
        # itself at verify time — unlike uv below, no observed task behavior
        # depends on their absence.)
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "--no-install-recommends curl ca-certificates bash",
        # uv + its own managed Python: immune to whatever python (if any)
        # the task base image ships. Installed OUTSIDE PATH (/opt/uv) so the
        # agent's PATH lookup stays faithful to the official task image: no
        # tool the image didn't ship resolves from PATH, and a task image's
        # own uv (e.g. financial-document-processor ships uv 0.8.14 at /bin)
        # is never shadowed. (Filesystem traces under /opt remain visible —
        # a single-container sandbox cannot hide them from a root agent.)
        "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh",
        "/opt/uv/uv venv --python 3.12 /opt/envserver",
        # THIS checkout's tbench2_env source (embedded), not a released
        # package: local fixes (canonical evaluate, TB2_COMMAND_TIMEOUT_S)
        # ship with the image. Deps (openenv, camel-ai, ...) come from PyPI.
        f"mkdir -p /opt/src && echo {_env_src_tar_b64()} | base64 -d | tar xz -C /opt/src",
        "/opt/uv/uv pip install --python /opt/envserver/bin/python /opt/src/tbench2_env_src uvicorn gradio",
        # Task directory for reset(task_id) via TB2_TASKS_DIR (pinned-SHA
        # GitHub tarball; see _task_layer_command).
        _task_layer_command(task_dir),
    ]


def build_task_image(task_dir: Path, docker_image: str | None = None):
    """Daytona-declarative expression of the same recipe (same layers, so the
    Daytona build cache is shared with the Dockerfile expression)."""
    from daytona import Image

    task_dir = Path(task_dir)
    base = _resolve_docker_image(task_dir, docker_image)
    return (
        Image.base(base).run_commands(*_server_layer_commands(task_dir))
        # Daytona does not execute the image CMD; a long-lived entrypoint keeps
        # the sandbox alive and the caller execs server_cmd() explicitly.
        .entrypoint(["sleep", "infinity"])
    )


def task_resources(task_dir: Path):
    from daytona import Resources

    env_cfg = read_task_config(task_dir).get("environment", {})
    return Resources(
        cpu=max(1, int(env_cfg.get("cpus", 1))),
        memory=max(2, int(env_cfg.get("memory_mb", 2048)) // 1024),
        disk=max(10, int(env_cfg.get("storage_mb", 10240)) // 1024),
    )


def wait_server_ready(base_url: str, timeout_s: float = 300.0) -> None:
    import requests

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(2.0)
    raise TimeoutError(f"env server at {base_url} not ready in {timeout_s}s ({last_err})")


def create_task_sandbox(
    daytona,
    task_dir: Path,
    *,
    command_timeout_s: int = 900,
    create_timeout_s: float = 1800.0,
    ready_timeout_s: float = 300.0,
):
    """Create ONE per-episode sandbox for *task_dir*, declaratively (no named snapshot).

    Returns ``(sandbox, base_url)``. Caller must ``daytona.delete(sandbox)``
    when the episode ends. First create for a task pays the image build;
    repeat creates hit Daytona's build cache.
    """
    from daytona import CreateSandboxFromImageParams

    params = CreateSandboxFromImageParams(
        image=build_task_image(task_dir),
        resources=task_resources(task_dir),
        auto_stop_interval=0,
        # Ownership marker: lets sweep/cleanup tooling target exactly the
        # sandboxes this recipe created (shared orgs run other workloads).
        labels={"openenv-tbench2-task": task_dir.name},
    )
    sandbox = daytona.create(params, timeout=create_timeout_s)
    try:
        cmd = server_cmd(command_timeout_s, default_task_id=task_dir.name)
        sandbox.process.exec(
            f"nohup bash -c {shlex.quote(cmd)} > /tmp/openenv-server.log 2>&1 &" " echo $! > /tmp/openenv-server.pid",
            timeout=10,
        )
        url = sandbox.create_signed_preview_url(8000, expires_in_seconds=86400).url
        wait_server_ready(url, timeout_s=ready_timeout_s)
        return sandbox, url
    except Exception:
        daytona.delete(sandbox)
        raise


def make_daytona():
    """Daytona client from the env-var contract (DAYTONA_API_KEY, optional
    DAYTONA_API_URL). Public: callers driving create_task_sandbox() need a
    client configured the same way this module's own CLI is."""
    from daytona import Daytona, DaytonaConfig

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise RuntimeError("DAYTONA_API_KEY not set")
    return Daytona(
        DaytonaConfig(
            api_key=api_key,
            api_url=os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api"),
        )
    )


def bake(daytona, tasks_dir: Path, task_id: str, prefix: str, force: bool) -> None:
    """Register the named snapshot ``<prefix><task-id>`` (optional warm cache)."""
    from daytona import CreateSnapshotParams

    task_dir = tasks_dir / task_id
    name = snapshot_name(prefix, task_id)
    try:
        existing = daytona.snapshot.get(name)
    except Exception:
        existing = None
    if existing is not None:
        if not force:
            print(f"[skip] {name} already exists (state={getattr(existing, 'state', '?')})")
            return
        print(f"[force] deleting existing {name}")
        daytona.snapshot.delete(existing)

    resources = task_resources(task_dir)
    print(f"[bake] {name}  cpu={resources.cpu} mem={resources.memory}G disk={resources.disk}G")
    daytona.snapshot.create(
        CreateSnapshotParams(
            name=name,
            image=build_task_image(task_dir),
            resources=resources,
            entrypoint=["sleep", "infinity"],
        ),
        on_logs=lambda line: print(f"  | {line}", flush=True),
        timeout=1800,
    )
    print(f"[done] {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-dir", required=True, help="local terminal-bench-2 checkout")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks", help="comma-separated task_ids")
    group.add_argument("--all", action="store_true", help="every dir with a task.toml")
    ap.add_argument("--prefix", default="tb2-", help="snapshot name prefix (default: tb2-)")
    ap.add_argument("--force", action="store_true", help="recreate existing snapshots")
    args = ap.parse_args()

    daytona = make_daytona()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    if args.all:
        task_ids = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task_id in task_ids:
        bake(daytona, tasks_dir, task_id, args.prefix, args.force)


if __name__ == "__main__":
    sys.exit(main())
