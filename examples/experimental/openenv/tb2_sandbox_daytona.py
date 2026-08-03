"""Daytona materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling module)
and is provider-agnostic. This module is everything Daytona-specific about
turning that recipe into a running cloud sandbox:

  ``create_task_sandbox(...)``  per-episode declarative create straight from
      the ``Image`` definition. Named snapshots count against an org-level
      quota, so registering one per task may not scale to a full task suite;
      the declarative path avoids the quota entirely, and repeat creates hit
      Daytona's build cache (~1min after the first build). Daytona does not
      run the image CMD, so this execs ``server_cmd()`` and waits for /health.
      Sandboxes carry ownership labels (task / launcher / run id) and an
      auto-stop+auto-delete TTL armed as a dead-man's switch: a keepalive
      thread beats the activity timer while the creating process lives, so
      a hard-killed caller's orphans are reclaimed instead of billing forever.

There is deliberately no bake step here: on this provider the image definition
IS the cache key, so a create either hits the build cache or warms it, and
nothing a create passes can name a pre-registered snapshot — registering one
per task would only spend the org quota the declarative path exists to avoid.
"""

import getpass
import os
import shlex
import threading
import time
from pathlib import Path

from tb2_sandbox_recipe import (
    read_task_config,
    resolve_docker_image,
    server_cmd,
    server_layer_commands,
    wait_server_ready,
)


def build_task_image(task_dir: Path, docker_image: str | None = None):
    """Daytona-declarative expression of the recipe (same layers as a
    Dockerfile expression would use, so the Daytona build cache is shared)."""
    from daytona import Image

    task_dir = Path(task_dir)
    base = resolve_docker_image(task_dir, docker_image)
    return (
        Image.base(base).run_commands(*server_layer_commands(task_dir))
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


def sandbox_labels(task_dir: Path) -> dict[str, str]:
    """Labels for a per-task sandbox: what it runs, and who launched it.

    The Daytona API records no creator, so in a shared org labels are the only
    attribution. ``openenv-tbench2-task`` keys sweep/cleanup tooling to exactly
    the sandboxes this recipe created (shared orgs run other workloads).
    ``openenv-launcher`` is OPENENV_LAUNCHER when set — do set it on shared
    hosts, where the unix user is a generic account — else the local unix
    user. ``openenv-run-id`` (OPENENV_RUN_ID, optional) additionally groups
    one run's sandboxes for targeted sweeps.
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


# Keepalive cadence: 6 beats per 30-minute auto-stop window, and up to 3
# consecutive failed beats (15 minutes of API blips) tolerated before the
# thread concludes the sandbox is gone and exits.
_KEEPALIVE_INTERVAL_S = 300.0
_KEEPALIVE_MAX_CONSECUTIVE_FAILURES = 3


def _start_keepalive(sandbox, task_id: str) -> None:
    """Refresh the sandbox's activity timer for as long as THIS process lives.

    Daytona's auto-stop clock counts only SDK interactions — preview-proxy
    traffic, which is ALL of an episode's I/O, does not reset it — so without
    a heartbeat any healthy episode longer than the auto-stop interval would
    be stopped mid-run. A daemon thread has exactly the right lifetime: it
    dies with the process, which is what turns auto-stop into a dead-man's
    switch for orphans. The thread exits once refreshes fail persistently
    (the normal case: the episode ended and the caller deleted the sandbox).
    """

    def _beat() -> None:
        failures = 0
        while failures < _KEEPALIVE_MAX_CONSECUTIVE_FAILURES:
            time.sleep(_KEEPALIVE_INTERVAL_S)
            try:
                sandbox.refresh_activity()
                failures = 0
            except Exception:
                failures += 1

    threading.Thread(target=_beat, name=f"tb2-sandbox-keepalive-{task_id}", daemon=True).start()


def create_task_sandbox(
    daytona,
    task_dir: Path,
    *,
    command_timeout_s: int = 900,
    create_timeout_s: float = 1800.0,
    ready_timeout_s: float = 300.0,
    auto_stop_minutes: int = 30,
    auto_delete_minutes: int = 120,
):
    """Create ONE per-episode sandbox for *task_dir*, declaratively (no named snapshot).

    Returns ``(sandbox, base_url)``. Caller must ``daytona.delete(sandbox)``
    when the episode ends. First create for a task pays the image build;
    repeat creates hit Daytona's build cache.

    Orphan TTL: a caller that dies without reaching its delete (SIGKILL, OOM,
    node loss) leaks the sandbox, and Daytona's defaults would keep it running
    — and billing — forever. Auto-stop/auto-delete arm a backstop, and a
    keepalive thread (see ``_start_keepalive``) beats the activity timer for
    as long as the creating process lives: a live episode of any length is
    safe, while a dead caller stops beating and Daytona stops the sandbox
    within *auto_stop_minutes* of the last beat, then deletes the stopped
    remains after *auto_delete_minutes* more.
    """
    from daytona import CreateSandboxFromImageParams

    params = CreateSandboxFromImageParams(
        image=build_task_image(task_dir),
        resources=task_resources(task_dir),
        auto_stop_interval=auto_stop_minutes,
        auto_delete_interval=auto_delete_minutes,
        labels=sandbox_labels(task_dir),
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
        _start_keepalive(sandbox, task_dir.name)
        return sandbox, url
    except Exception:
        daytona.delete(sandbox)
        raise


_DEFAULT_API_KEY_FILE = "~/.config/daytona/api_key"


def resolve_api_key() -> str:
    """The Daytona API key: DAYTONA_API_KEY, else the key file.

    The file indirection (DAYTONA_API_KEY_FILE, default
    ``~/.config/daytona/api_key``) exists so launchers can hand rollout
    workers a PATH instead of the secret itself: anything a launcher
    forwards rides ray's runtime_env, which is echoed into driver logs and
    persisted in job metadata in plaintext. Env vars the worker already has
    (platform-injected, single-host inheritance) never pass through ray, so
    DAYTONA_API_KEY is checked first.
    """
    key = os.environ.get("DAYTONA_API_KEY", "").strip()
    if key:
        return key
    key_file = Path(os.environ.get("DAYTONA_API_KEY_FILE", "").strip() or _DEFAULT_API_KEY_FILE).expanduser()
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    if not key:
        raise RuntimeError(f"no Daytona API key: DAYTONA_API_KEY is unset and {key_file} is missing or empty")
    return key


def make_daytona():
    """Daytona client: key from resolve_api_key(), endpoint from optional
    DAYTONA_API_URL. Public: callers driving create_task_sandbox() need a
    client configured this way."""
    from daytona import Daytona, DaytonaConfig

    return Daytona(
        DaytonaConfig(
            api_key=resolve_api_key(),
            api_url=os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api"),
        )
    )
