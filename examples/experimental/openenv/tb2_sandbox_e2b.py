"""E2B materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling
module) and is provider-agnostic. This module is everything E2B-specific about
turning that recipe into a running cloud sandbox, and it works against any
server speaking the E2B API:

  * **E2B Cloud** (the default the SDK ships with), or
  * **a self-hosted `AgentENV <https://github.com/kvcache-ai/AgentENV>`_
    deployment** — the Firecracker microVM platform whose native API *is* the
    E2B API. Point ``E2B_API_URL`` (and ``E2B_SANDBOX_URL``) at the AgentENV
    server and this module needs no code changes; AgentENV currently accepts
    any non-empty API key.

Unlike Daytona's declarative per-episode image build, E2B separates the two
halves of a create:

  ``ensure_task_template(...)``  build the per-task template once, under a
      deterministic alias derived from the task id AND a digest of the recipe
      itself — so editing the recipe re-bakes automatically instead of
      silently serving stale templates. Template counts are not quota-bound
      (self-hosted AgentENV: bound only by snapshot-store capacity).
  ``create_task_sandbox(...)``  per-episode: warm-start a sandbox from the
      template, exec ``server_cmd()``, wait for /health, return
      ``(sandbox, base_url)``. The sandbox carries ownership metadata and a
      TTL armed as a dead-man's switch: a keepalive thread re-arms the
      timeout while the creating process lives, so a hard-killed caller's
      orphans are reclaimed instead of running forever.
  bake CLI (``python tb2_sandbox_e2b.py ...``)  pre-build templates for a
      task list so the first training episode doesn't pay the image build.

All e2b imports are lazy (in-function), mirroring ``tb2_sandbox_daytona``:
offline unit tests and non-sandbox launches must not require the SDK.
"""

import argparse
import concurrent.futures
import hashlib
import os
import re
import shlex
import sys
import threading
from pathlib import Path

import tb2_sandbox_recipe as recipe
from tb2_sandbox_recipe import (
    read_task_config,
    resolve_docker_image,
    sandbox_labels,
    server_cmd,
    server_layer_commands,
    wait_server_ready,
)


def template_alias(task_dir: Path) -> str:
    """Deterministic template alias: ``tb2-<task-id>-<recipe digest>``.

    The digest covers the base image, every build command (which embed the
    tbench2_env source, deterministically tarred — see ``_dir_tar_b64``), and
    the build resources (E2B sizes sandboxes at template-build time), so the
    alias changes exactly when the baked artifact would: recipe edits,
    env-package changes, or a task.toml resource bump re-bake; identical
    inputs reuse the existing template.
    """
    task_dir = Path(task_dir)
    base = resolve_docker_image(task_dir, None)
    resources = task_build_resources(task_dir)
    inputs = [base, *server_layer_commands(task_dir), repr(sorted(resources.items()))]
    digest = hashlib.sha256("\n".join(inputs).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9-]", "-", task_dir.name.lower())
    return f"tb2-{slug}-{digest}"


def task_build_resources(task_dir: Path) -> dict[str, int]:
    """Template build resources from ``task.toml [environment]``.

    E2B sizes sandboxes at template-build time (warm starts inherit the
    template's spec), so the task's requirements go here, not on create.
    """
    env_cfg = read_task_config(task_dir).get("environment", {})
    return {
        "cpu_count": max(1, int(env_cfg.get("cpus", 1))),
        "memory_mb": max(2048, int(env_cfg.get("memory_mb", 2048))),
    }


def _connection_opts() -> dict:
    """Per-call connection kwargs. Only the key is passed explicitly (for the
    file indirection); endpoint overrides (E2B_API_URL, E2B_SANDBOX_URL,
    E2B_DOMAIN — e.g. a self-hosted AgentENV) are read from the environment
    by the SDK itself."""
    return {"api_key": resolve_api_key()}


# One build per alias per process: a rollout fans out many episodes of the
# same task at once, and every concurrent miss would otherwise start its own
# multi-minute build. Cross-process dedup is the server's job (build cache).
_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def _build_lock(alias: str) -> threading.Lock:
    with _build_locks_guard:
        return _build_locks.setdefault(alias, threading.Lock())


def ensure_task_template(
    task_dir: Path,
    *,
    force: bool = False,
    build_timeout_s: float = 1800.0,
    on_logs=None,
) -> str:
    """Build the per-task template if its alias doesn't exist yet; return the alias.

    *build_timeout_s* bounds the build's wall clock. ``Template.build`` blocks
    across many requests (trigger + log polling), so the deadline is enforced
    by running it on a scoped thread; on timeout the provider-side build keeps
    cooking (and may finish), but this caller stops holding the alias lock and
    the create-semaphore slot.
    """
    # In-function import, deliberately: the e2b SDK is an optional dependency
    # of this recipe (the launcher preflights it), so importing the module
    # must not require it — the offline tests import this file with a fake
    # `e2b` in sys.modules, and the Daytona leg defers its SDK the same way.
    from e2b import Template

    task_dir = Path(task_dir)
    alias = template_alias(task_dir)
    with _build_lock(alias):
        if not force and Template.alias_exists(alias, **_connection_opts()):
            return alias
        base = resolve_docker_image(task_dir, None)
        template = Template().from_image(base)
        for command in server_layer_commands(task_dir):
            template = template.run_cmd(command)

        def _build() -> None:
            Template.build(
                template,
                alias,
                **task_build_resources(task_dir),
                skip_cache=force,
                on_build_logs=on_logs,
                **_connection_opts(),
            )

        # Not a `with` block: its __exit__ joins the worker, which would wait
        # out the very build the timeout is meant to abandon.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            pool.submit(_build).result(timeout=build_timeout_s)
        finally:
            pool.shutdown(wait=False)
    return alias


def base_url(sandbox) -> str:
    """The env server's externally reachable URL on port 8000.

    ``get_host`` yields a per-port hostname routed by the provider's gateway
    (E2B Cloud, or AgentENV's gateway ``{port}-{sandboxID}.{domain}`` routing
    — the deployment must have a routing domain configured). The gateway must
    proxy WebSocket as well as HTTP: OpenEnv's client health-checks over HTTP
    but runs episode I/O over ``/ws``. OPENENV_E2B_URL_SCHEME (default https)
    covers plain-HTTP self-hosted gateways.
    """
    scheme = os.getenv("OPENENV_E2B_URL_SCHEME", "https")
    return f"{scheme}://{sandbox.get_host(8000)}"


# TTL / keepalive: the sandbox is created with a bounded lifetime and the
# keepalive thread re-arms it ahead of expiry for as long as THIS process
# lives — the E2B timeout is kill-on-expiry, which is exactly a dead-man's
# switch for orphans. 6 beats per window; up to 3 consecutive failed beats
# (API blips) tolerated before the thread concludes the sandbox is gone.
_SANDBOX_TTL_S = int(os.getenv("OPENENV_E2B_SANDBOX_TTL_S", "1800"))
_KEEPALIVE_INTERVAL_S = _SANDBOX_TTL_S / 6.0
_KEEPALIVE_MAX_CONSECUTIVE_FAILURES = 3


def _start_keepalive(sandbox, task_id: str) -> None:
    """Re-arm the sandbox TTL for as long as THIS process lives.

    A live episode of any length keeps its sandbox; a hard-killed caller
    (SIGKILL, OOM, node loss) stops beating and the provider kills the
    sandbox within _SANDBOX_TTL_S of the last beat. The thread exits once
    re-arms fail persistently (the normal case: the episode ended and the
    caller killed the sandbox)."""
    opts = _connection_opts()
    recipe.start_keepalive(
        lambda: sandbox.set_timeout(_SANDBOX_TTL_S, **opts),
        f"tb2-sandbox-keepalive-{task_id}",
        interval_s=_KEEPALIVE_INTERVAL_S,
        max_consecutive_failures=_KEEPALIVE_MAX_CONSECUTIVE_FAILURES,
    )


def create_task_sandbox(
    task_dir: Path,
    *,
    command_timeout_s: int = 900,
    ready_timeout_s: float = 300.0,
    ttl_s: int | None = None,
):
    """Create ONE per-episode sandbox for *task_dir* from its template.

    Returns ``(sandbox, base_url)``. Caller must ``sandbox.kill()`` when the
    episode ends. First create for a task pays the template build (via
    ``ensure_task_template``); repeats warm-start from the built template.

    The template deliberately carries no start command: the server is exec'd
    here so runtime knobs (TB2_COMMAND_TIMEOUT_S) stay runtime — baking them
    would silently pin them until the next re-bake.
    """
    from e2b import Sandbox

    task_dir = Path(task_dir)
    opts = _connection_opts()
    alias = ensure_task_template(task_dir)
    sandbox = Sandbox.create(
        template=alias,
        timeout=ttl_s if ttl_s is not None else _SANDBOX_TTL_S,
        metadata=sandbox_labels(task_dir),
        **opts,
    )
    try:
        cmd = server_cmd(command_timeout_s, default_task_id=task_dir.name)
        sandbox.commands.run(
            f"bash -c {shlex.quote(cmd)} > /tmp/openenv-server.log 2>&1",
            background=True,
        )
        url = base_url(sandbox)
        wait_server_ready(url, timeout_s=ready_timeout_s)
        _start_keepalive(sandbox, task_dir.name)
        return sandbox, url
    except Exception:
        sandbox.kill(**opts)
        raise


def kill_sandbox(sandbox) -> None:
    """Kill a sandbox created here.

    The handle does not carry the endpoint and key its create used, so they
    have to be supplied again — which is this module's business, not its
    callers'.
    """
    sandbox.kill(**_connection_opts())


_DEFAULT_API_KEY_FILE = "~/.config/e2b/api_key"


def resolve_api_key() -> str:
    """The E2B API key: E2B_API_KEY, else the key file (see
    recipe.resolve_api_key for the file-indirection rationale). AgentENV does
    not enforce keys today, but recent SDKs validate the format client-side —
    provision a well-formed one (e2b_ + 40 hex chars)."""
    return recipe.resolve_api_key("E2B_API_KEY", "E2B_API_KEY_FILE", _DEFAULT_API_KEY_FILE)


def bake(tasks_dir: Path, task_id: str, force: bool) -> None:
    """Pre-build the template for one task (optional warm cache)."""
    task_dir = tasks_dir / task_id
    alias = template_alias(task_dir)
    resources = task_build_resources(task_dir)
    print(f"[bake] {alias}  cpu={resources['cpu_count']} mem={resources['memory_mb']}MB")
    ensure_task_template(
        task_dir,
        force=force,
        on_logs=lambda entry: print(f"  | {entry}", flush=True),
    )
    print(f"[done] {alias}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    recipe.add_task_selection_args(ap)
    ap.add_argument("--force", action="store_true", help="rebuild even when the alias exists")
    args = ap.parse_args()

    tasks_dir, task_ids = recipe.selected_task_ids(args)

    for task_id in task_ids:
        bake(tasks_dir, task_id, args.force)


if __name__ == "__main__":
    sys.exit(main())
