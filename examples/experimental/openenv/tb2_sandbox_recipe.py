"""Provider-agnostic image recipe for per-task Terminal-Bench-2 sandboxes.

TB2 is a per-task-image benchmark: every task pins its official runtime image
in ``task.toml`` (``[environment].docker_image`` — the exact image the TB2
harness itself runs). A sandbox serving this env must therefore be built per
task: **the official task image ⊕ this env's server layer**, one layer, no
DinD. This module owns that recipe as plain shell commands — nothing
provider-specific — so the same layers can back a Dockerfile, a Daytona
declarative build (``tb2_sandbox_daytona``, the sibling module), or another
provider.

  ``server_layer_commands(task_dir)``  shell commands that turn the official
      task image into a combined task+env-server image: a uv-managed Python
      venv at ``/opt/envserver`` running the INSTALLED tbench2_env package
      (source embedded into the build, so the sandbox runs exactly the
      version installed here — no released package needed), plus the task
      directory staged at ``/opt/tb2-tasks/<id>``
      (downloaded from the tasks checkout's pinned-commit GitHub tarball) for
      ``reset(task_id)`` via ``TB2_TASKS_DIR``.
  ``server_cmd()``  starts the env server inside the sandbox. Sets
      ``CAMEL_RUNTIME=true`` so camel's TerminalToolkit runs commands in the
      task image's native environment instead of hijacking PATH with its own
      Python 3.10 ``.initial_env`` venv (real TB2 agents see the image's
      python), and ``TB2_COMMAND_TIMEOUT_S`` for realistic command budgets.

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

import base64
import concurrent.futures
import getpass
import gzip
import io
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

# Guard for the ONE payload that must be embedded into the build: the
# tbench2_env package source (embedding the local install is what guarantees
# the sandbox scores with exactly the version validated here).
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

# Per-exec timeout inside the sandbox. Named after the variable the env server
# itself reads (it is tbench2_env's contract, not ours), and owned here because
# this is where the server's command line is built — every backend gets the same
# value without three copies of this getenv.
COMMAND_TIMEOUT_S = int(os.getenv("TB2_COMMAND_TIMEOUT_S", "900"))

# The command to start the env server inside a task sandbox (a provider may
# not run the image CMD — Daytona doesn't). /opt/envserver and /opt/tb2-tasks
# are baked by server_layer_commands.
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


def server_cmd(command_timeout_s: int = COMMAND_TIMEOUT_S, default_task_id: str = "") -> str:
    # A per-task sandbox stages exactly one task, so make it the default:
    # a reset() with no task_id resolves to the staged task rather than the
    # env's built-in headless-terminal default (which isn't present here).
    return SERVER_CMD.format(
        command_timeout_s=command_timeout_s,
        default_task_id=shlex.quote(default_task_id),
    )


def read_task_config(task_dir: Path) -> dict:
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"{task_dir}: no task.toml")
    return tomllib.loads(toml_path.read_text())


def task_env_resources(task_dir: Path) -> tuple[int, int, int]:
    """``(cpus, memory_mb, storage_mb)`` from ``task.toml [environment]``, floored.

    The floors (1 CPU / 2048 MB memory / 10240 MB disk) are recipe policy —
    the env server needs room to run alongside the task — applied once here so
    the providers cannot drift apart on them. Each materialization maps these
    onto its own units and drops what its provider does not take (Modal sizes
    disk itself; E2B sizes everything at template-build time).
    """
    env_cfg = read_task_config(task_dir).get("environment", {})
    return (
        max(1, int(env_cfg.get("cpus", 1))),
        max(2048, int(env_cfg.get("memory_mb", 2048))),
        max(10240, int(env_cfg.get("storage_mb", 10240))),
    )


def sandbox_labels(task_dir: Path) -> dict[str, str]:
    """Ownership labels/metadata for a per-task sandbox: what it runs, and who
    launched it. Provider-agnostic (Daytona labels, E2B metadata, Modal tags —
    same keys).

    Sandbox APIs record no creator, so in a shared org/deployment these are
    the only attribution. ``openenv-tbench2-task`` keys sweep/cleanup tooling
    to exactly the sandboxes this recipe created (shared orgs run other
    workloads). ``openenv-launcher`` is OPENENV_LAUNCHER when set — do set it
    on shared hosts, where the unix user is a generic account — else the local
    unix user. ``openenv-run-id`` (OPENENV_RUN_ID, optional) additionally
    groups one run's sandboxes for targeted sweeps.
    """
    try:
        user = getpass.getuser()
    except Exception:  # no passwd entry / login env on minimal hosts
        user = "unknown"
    labels = {
        "openenv-tbench2-task": task_dir.name,
        "openenv-launcher": os.environ.get("OPENENV_LAUNCHER") or user,
    }
    run_id = os.environ.get("OPENENV_RUN_ID")
    if run_id:
        labels["openenv-run-id"] = run_id
    return labels


def start_keepalive(beat, thread_name: str, *, interval_s: float, max_consecutive_failures: int) -> None:
    """Run *beat* every *interval_s* for as long as THIS process lives.

    A daemon thread has exactly the right lifetime for a sandbox keepalive: it
    dies with the process, which is what turns the provider's idle/TTL timer
    into a dead-man's switch for orphans. The thread exits once beats fail
    *max_consecutive_failures* times in a row (the normal case: the episode
    ended and the caller deleted the sandbox).
    """

    def _run() -> None:
        failures = 0
        while failures < max_consecutive_failures:
            time.sleep(interval_s)
            try:
                beat()
                failures = 0
            except Exception:
                failures += 1

    threading.Thread(target=_run, name=thread_name, daemon=True).start()


def run_with_deadline(fn, timeout_s: float):
    """Run *fn* on a scoped worker thread; raise TimeoutError past *timeout_s*.

    For provider calls that block across many requests with no deadline of
    their own (an E2B template build, a Modal build+create). Deliberately not
    a ``with ThreadPoolExecutor`` block: its ``__exit__`` joins the worker,
    which would wait out the very call the deadline exists to abandon. On
    timeout the provider-side work keeps cooking (and may finish) — reclaiming
    whatever it produces is the caller's provider-specific business — but this
    caller stops holding its locks and semaphore slots.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


def add_task_selection_args(ap) -> None:
    """The bake CLIs' shared task selection: --tasks-dir + (--tasks | --all)."""
    ap.add_argument("--tasks-dir", required=True, help="local terminal-bench-2 checkout")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks", help="comma-separated task_ids")
    group.add_argument("--all", action="store_true", help="every dir with a task.toml")


def selected_task_ids(args) -> tuple[Path, list[str]]:
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    if args.all:
        task_ids = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    return tasks_dir, task_ids


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = Path(tarinfo.name).name
    if name in {"__pycache__", ".initial_env"} or name.endswith((".pyc", ".egg-info")):
        return None
    # Zero out metadata the build doesn't need (mtimes, owner): it varies
    # across hosts and checkouts and would break _dir_tar_b64's determinism.
    tarinfo.mtime = 0
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""
    return tarinfo


def _dir_tar_b64(paths: list[Path], arcnames: list[str], max_bytes: int) -> str:
    # Must be byte-for-byte deterministic for identical source: the b64 is
    # embedded in a build command, so any drift — the gzip header's
    # compression timestamp (mtime=0 suppresses it), per-entry mtimes/owners
    # (_tar_filter zeroes them) — would change the image definition on every
    # call and defeat provider build caches and pre-baked snapshots.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
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
    payload-free, so build commands stay far below provider build-command
    size ceilings (see _MAX_INLINE_TAR_BYTES) regardless of task-dir size.
    Requires the tasks checkout to be a git clone with a GitHub origin.
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
    # In-function, deliberately: only a build that embeds the env source needs
    # the package, and the offline tests import this module without it installed.
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


def resolve_docker_image(task_dir: Path, docker_image: str | None) -> str:
    if docker_image:
        return docker_image
    docker_image = read_task_config(task_dir).get("environment", {}).get("docker_image")
    if not docker_image:
        raise ValueError(f"{task_dir.name}: task.toml has no [environment].docker_image")
    return docker_image


def server_layer_commands(task_dir: Path) -> list[str]:
    """The recipe itself: shell commands that turn the OFFICIAL task image
    into a combined task+env-server image. Nothing here is provider-specific —
    the same layers work for docker build, Daytona, Modal, ACA, etc.
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


def wait_server_ready(base_url: str, timeout_s: float = 300.0) -> None:
    # In-function, deliberately: requests is not a hard dependency of this
    # module -- the offline tests import it with requests absent -- and only a
    # caller that actually waits on a live server needs it.
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
