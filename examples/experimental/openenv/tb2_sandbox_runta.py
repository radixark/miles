"""Runta materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling
module) and is provider-agnostic. This module is everything Runta-specific
about turning that recipe into a running cloud sandbox.

A Runta runtime is a base machine, not a container started from an image:
``runtimes.create`` takes no image, and the base machine ships Docker. So the
recipe is expressed as a Dockerfile and built by the Docker daemon *inside* a
runtime, and the env server runs as the one container of that image — the
agent still gets exactly the task image's filesystem, user and working
directory. What makes that affordable per episode is the checkpoint:

  ``ensure_task_checkpoint(...)``  build the per-task image once, in a
      throwaway builder runtime, and snapshot that runtime as a checkpoint
      named after the task id AND a digest of the recipe itself — so editing
      the recipe re-bakes automatically instead of silently serving a stale
      image. The checkpoint is Runta's analogue of an E2B template.
  ``create_task_sandbox(...)``  per-episode: restore a runtime from the
      checkpoint (the built image is already on its disk, nothing is pulled),
      ``docker run`` the env server, wait for /health through the published
      port, return ``(runtime, base_url)``. The runtime is created with a
      bounded lifetime armed as a dead-man's switch: a keepalive thread
      re-arms it while the creating process lives, so a hard-killed caller's
      orphans are shut down within ten minutes of its last beat, and deleted
      ten minutes after that, instead of running forever.
  ``delete_task_sandbox(...)``  the close of the above: stop the keepalive,
      then delete the runtime. The order is the point — see _Keepalive.
  bake CLI (``python tb2_sandbox_runta.py ...``)  pre-build checkpoints for a
      task list so the first training episode doesn't pay the image build.

Runta has no labels or metadata, so a runtime's name is its only attribution:
every runtime this module creates is named ``miles-tb2-...`` (``runta ps -a``
shows them), with ``OPENENV_RUN_ID`` folded in when set.

All runta imports are lazy (in-function), mirroring ``tb2_sandbox_e2b``:
offline unit tests and non-sandbox launches must not require the SDK.
"""

import argparse
import hashlib
import os
import re
import shlex
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import tb2_sandbox_recipe as recipe
from tb2_sandbox_recipe import (
    resolve_docker_image,
    server_cmd,
    server_layer_commands,
    task_env_resources,
    wait_server_ready,
)
from miles.rollout.agentic.credentials import resolve_provider_api_key
from miles.rollout.agentic.rollout_limits import rollout_lock


# The user every build command and the env server run as. The TB2 task images
# are built for a root agent (their solutions and tests apt-install freely), so
# anything less would change the task environment, not just the build. It is
# written into the Dockerfile ahead of the first layer, which makes it the built
# image's user and therefore the user `docker run` starts the env server as.
_BUILD_USER = "root"

# Where the recipe lives inside a runtime: the built image's tag, the one
# container of it, and the build context holding the generated Dockerfile.
_IMAGE_TAG = "tb2-env:latest"
_CONTAINER_NAME = "tb2-env"
_BUILD_DIR = "/opt/tb2-build"
_SERVER_PORT = 8000

# The Docker daemon lives in the runtime, outside the task container, so the
# runtime is sized above what task.toml asks for by what the daemon itself
# holds; the container then has the task's own figure to itself.
_DOCKER_OVERHEAD_MIB = 512

_NAME_PREFIX = "miles-tb2"


def task_slug(task_dir: Path) -> str:
    return re.sub(r"[^a-z0-9-]", "-", Path(task_dir).name.lower())


def task_runtime_resources(task_dir: Path) -> dict[str, int]:
    """Runtime size from ``task.toml [environment]``, in ``runtimes.create`` terms.

    Disk is not passed: a runtime's own disk (32 GiB on Runta Cloud) already
    exceeds the recipe's 10 GiB floor, which is also what every current TB2
    task asks for.
    """
    cpus, memory_mb, _storage_mb = task_env_resources(task_dir)
    return {"vcpus": cpus, "memory_mib": memory_mb + _DOCKER_OVERHEAD_MIB}


def dockerfile(task_dir: Path) -> str:
    """The recipe as a Dockerfile: the official task image plus one ``RUN`` per layer command."""
    base = resolve_docker_image(task_dir, None)
    lines = [f"FROM {base}", f"USER {_BUILD_USER}", *(f"RUN {command}" for command in server_layer_commands(task_dir))]
    return "\n".join(lines) + "\n"


def checkpoint_name(task_dir: Path) -> str:
    """Deterministic checkpoint name: ``tb2-<task-id>-<recipe digest>``.

    The digest covers the Dockerfile (base image, build user, and every build
    command, which embed the tbench2_env source, deterministically tarred — see
    ``_dir_tar_b64``) and the runtime's size (a checkpoint is a snapshot of a
    sized machine, and episodes restore it at that size), so the name changes
    exactly when the baked artifact would: recipe edits, env-package changes,
    or a task.toml resource bump re-bake; identical inputs reuse the existing
    checkpoint.
    """
    task_dir = Path(task_dir)
    resources = task_runtime_resources(task_dir)
    inputs = [dockerfile(task_dir), _IMAGE_TAG, repr(sorted(resources.items()))]
    digest = hashlib.sha256("\n".join(inputs).encode()).hexdigest()[:10]
    return f"tb2-{task_slug(task_dir)}-{digest}"


def _runtime_name(purpose: str, task_dir: Path) -> str:
    """``miles-tb2-<purpose>[-<run id>]-<task>-<random>``: the name is the only
    place ownership can go, so it carries what the other providers put in labels."""
    run_id = re.sub(r"[^a-z0-9-]", "-", os.environ.get("OPENENV_RUN_ID", "").lower())
    parts = [_NAME_PREFIX, purpose, run_id, task_slug(task_dir)[:40], uuid.uuid4().hex[:8]]
    return "-".join(part for part in parts if part)


# TTL / keepalive. auto-shutdown is a limit on one running session, and saving
# it again restarts its clock — which is exactly a dead-man's switch for
# orphans: the keepalive thread re-arms it ahead of expiry for as long as THIS
# process lives. A hard-killed caller's runtime therefore keeps running, and
# billing, for at most the TTL after its last beat, so the TTL is sized to the
# gap between beats and not to an episode: ten minutes, where a killed
# 10-runtime rollout was seen to leave 3 running towards a 30-minute one.
#
# 5 beats per window. A beat that fails outright costs one interval, so 3 in a
# row (API blips, a revision conflict) still leave the 4th to land inside the
# window; a slow one has the rest of the window to land in, which at the
# default is the 8 minutes the SDK will go on retrying a PATCH for. The thread
# gives up only after a full window of consecutive failures, by when the
# runtime has reached its TTL whatever the next beat does. A runtime that does
# get shut down is deleted _DELETE_AFTER_S later, so an orphan stops costing
# compute at the TTL and storage shortly after.
_SANDBOX_TTL_S = int(os.getenv("OPENENV_RUNTA_SANDBOX_TTL_S", "600"))
_READY_TIMEOUT_S = float(os.getenv("OPENENV_RUNTA_READY_TIMEOUT_S", "300"))
_KEEPALIVE_BEATS_PER_TTL = 5
_KEEPALIVE_INTERVAL_S = _SANDBOX_TTL_S / _KEEPALIVE_BEATS_PER_TTL
_KEEPALIVE_MAX_CONSECUTIVE_FAILURES = _KEEPALIVE_BEATS_PER_TTL
# How long a delete waits for a beat that is in flight when it stops the
# keepalive. A beat is one GET and one PATCH; only one stuck in the SDK's
# retries outlasts this, and the delete's own conflict retry covers that one.
_KEEPALIVE_STOP_TIMEOUT_S = 30.0
_DELETE_AFTER_S = 600
_CREATE_TIMEOUT_S = 240.0
_CHECKPOINT_TIMEOUT_S = 900.0


_DEFAULT_TOKEN_FILE = "~/.config/runta/token"


def resolve_token() -> str:
    """The Runta token: RUNTA_TOKEN, else the token file (see
    resolve_provider_api_key for the file-indirection rationale)."""
    return resolve_provider_api_key("RUNTA_TOKEN", "RUNTA_TOKEN_FILE", _DEFAULT_TOKEN_FILE)


_client_lock = threading.Lock()
_client = None


def make_client():
    """The process's Runta client: token from resolve_token(), endpoint left to
    the SDK's own resolution (api_endpoint reports what that comes to, and
    relies on none being passed here).

    Built once and shared: the client owns a connection pool, and this runs
    once per sandbox-create attempt, retries included, so a client per call
    turns a spell of API failures into a socket leak.
    """
    global _client
    with _client_lock:
        if _client is None:
            # In-function import, deliberately: the runta SDK is an optional
            # dependency of this recipe (the launcher preflights it), so
            # importing the module must not require it — the offline tests
            # import this file with a fake `runta` in sys.modules.
            from runta import Runta

            _client = Runta(token=resolve_token())
        return _client


def _create_runtime(client, name: str, **create_kwargs):
    """Create a runtime and wait for it, deleting it if it never becomes ready.

    ``wait=False`` is what makes that possible: a create that waits and then
    times out raises without ever returning the handle its delete needs.
    """
    runtime = client.runtimes.create(name, wait=False, **create_kwargs)
    try:
        runtime.wait_until_ready(timeout=_CREATE_TIMEOUT_S)
    except Exception:
        _delete_quietly(runtime)
        raise
    return runtime


class _Keepalive:
    """One runtime's keepalive thread, and the stop its delete needs.

    recipe.start_keepalive, which the other backends share, ends only by
    failing: the caller deletes the sandbox and the beats that follow find it
    gone. That cannot be left to chance here. Runta's delete is a GET of the
    runtime's revision and a DELETE expecting that revision, a beat is a PATCH
    that moves it, and the SDK does not retry a delete that lost that race — so
    a beat landing between the two turned the delete into a logged 409 and left
    the runtime running until its TTL. Hence a thread that can be told to stop
    and waited for: the interval is a wait on the stop event rather than a
    sleep, so stop() ends it at once, and stop() returns only when no beat is
    in flight (or _KEEPALIVE_STOP_TIMEOUT_S has passed).

    A daemon thread, as there: it dies with the process, which is what makes
    the TTL a dead-man's switch.
    """

    def __init__(self, beat, thread_name: str, *, interval_s: float, max_consecutive_failures: int):
        self._beat = beat
        self._interval_s = interval_s
        self._max_consecutive_failures = max_consecutive_failures
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        failures = 0
        # wait() is the sleep between beats, and True the moment stop() is called
        while failures < self._max_consecutive_failures and not self._stopped.wait(self._interval_s):
            try:
                self._beat()
                failures = 0
            except Exception:
                failures += 1

    def stop(self, timeout_s: float = _KEEPALIVE_STOP_TIMEOUT_S) -> None:
        self._stopped.set()
        self._thread.join(timeout_s)


# The keepalive of each runtime that has one, by runtime id: what lets a delete
# stop the thread given nothing but the runtime it is deleting.
_keepalives_lock = threading.Lock()
_keepalives: dict[str, _Keepalive] = {}


def _stop_keepalive(runtime) -> None:
    with _keepalives_lock:
        keepalive = _keepalives.pop(runtime.id, None)
    if keepalive is not None:
        keepalive.stop()


def _is_revision_conflict(exc: BaseException) -> bool:
    """True for the 409 a delete gets when the runtime changed under it.

    Read off the status, as the throttle classification is
    (openenv_runta_agent_function): the SDK raises one ApiError for every API
    failure. Imported here because this only runs once a delete has raised, and
    guarded so that classifying a failure can never replace it with another.
    """
    try:
        from runta import ApiError
    except ImportError:  # without the SDK nothing raised is a Runta API error
        return False
    return isinstance(exc, ApiError) and exc.status_code == 409


def delete_task_sandbox(runtime) -> None:
    """Delete a runtime this module created, its keepalive stopped first.

    The close of create_task_sandbox, and what every delete here goes through.
    With the keepalive stopped nothing in this process writes the runtime any
    more, but something else still can between the delete's GET and its DELETE:
    Runta itself (a status change moves the revision too), or a beat that
    outlasted the bounded wait. One such write is one conflict, so the delete
    is tried once more — the SDK re-reads the revision on every call — and a
    second conflict is raised like any other failure.
    """
    _stop_keepalive(runtime)
    try:
        runtime.delete()
    except Exception as exc:
        if not _is_revision_conflict(exc):
            raise
        runtime.delete()


def _delete_quietly(runtime) -> None:
    try:
        delete_task_sandbox(runtime)
    except Exception:
        pass  # cleanup must not mask the real failure; the TTL reclaims it


def _checked_exec(runtime, command: str, *, timeout: float, what: str) -> None:
    result = runtime.exec(command, timeout=timeout)
    if result.exit_code != 0:
        tail = (result.stdout_text + result.stderr_text)[-2000:]
        raise RuntimeError(f"{what} failed in runtime {runtime.id} (exit {result.exit_code}): {tail}")


def _ready_checkpoint_id(client, name: str) -> str | None:
    """The id of the usable checkpoint called *name*, or None when there is none.

    One still being created — by another rollout, since this one holds the
    build lock — is waited for rather than built a second time; a failed one
    is deleted so it can neither shadow nor collide with its replacement.
    """
    for checkpoint in client.checkpoints.list():
        if checkpoint.display_name != name:
            continue
        # CheckpointState is a str enum, so its members compare equal to their values
        if checkpoint.state == "ready":
            return checkpoint.id
        if checkpoint.state == "creating":
            return client.checkpoints.wait_until_ready(checkpoint.id, timeout=_CHECKPOINT_TIMEOUT_S).id
        client.checkpoints.delete(checkpoint.id)
    return None


def _build_checkpoint(client, task_dir: Path, name: str, *, build_timeout_s: float, on_logs=None) -> str:
    """Build the task image in a throwaway runtime and checkpoint that runtime."""
    log = on_logs or (lambda line: None)
    builder = _create_runtime(
        client,
        _runtime_name("build", task_dir),
        **task_runtime_resources(task_dir),
        # No keepalive: a build is bounded by its own deadlines, so the backstop
        # only has to outlast them.
        auto_shutdown_after_secs=int(_CREATE_TIMEOUT_S + build_timeout_s + _CHECKPOINT_TIMEOUT_S),
        auto_delete_after_secs=_DELETE_AFTER_S,
    )
    try:
        log(f"builder runtime {builder.id}")
        builder.files.write(f"{_BUILD_DIR}/Dockerfile", dockerfile(task_dir))
        # The build's output goes to a file and only its tail comes back, on
        # failure: exec results are buffered whole, and a build log is long.
        build_log = f"{_BUILD_DIR}/build.log"
        _checked_exec(
            builder,
            f"docker build -t {_IMAGE_TAG} {_BUILD_DIR} > {build_log} 2>&1 || {{ tail -c 4000 {build_log}; exit 1; }}",
            timeout=build_timeout_s,
            what="docker build",
        )
        log(f"image built; checkpointing as {name}")
        return builder.checkpoints.create(name=name, timeout=_CHECKPOINT_TIMEOUT_S).id
    finally:
        _delete_quietly(builder)


def ensure_task_checkpoint(
    task_dir: Path,
    *,
    force: bool = False,
    build_timeout_s: float = 1800.0,
    on_logs=None,
) -> str:
    """Build the per-task checkpoint if none exists under its name; return its id.

    *build_timeout_s* bounds the image build; the builder runtime's create and
    the checkpoint itself carry their own deadlines, so no call here blocks
    without one.
    """
    task_dir = Path(task_dir)
    client = make_client()
    name = checkpoint_name(task_dir)
    # one build per name per rollout: a rollout fans out many episodes of the same task at once,
    # and every concurrent miss would otherwise start its own multi-minute build
    with rollout_lock(f"openenv-runta-build-{name}"):
        checkpoint_id = _ready_checkpoint_id(client, name)
        if checkpoint_id and not force:
            return checkpoint_id
        if checkpoint_id:
            client.checkpoints.delete(checkpoint_id)
        return _build_checkpoint(client, task_dir, name, build_timeout_s=build_timeout_s, on_logs=on_logs)


# Where Runta Cloud serves published ports, keyed by its API host. The SDK
# resolves the API endpoint but exposes no helper for the ingress URL, so the
# rule Runta's own client uses is repeated in ingress.
_HOSTED_INGRESS_DOMAINS = {"api.runta.com": "runta.dev", "api.runta.me": "runta.sh"}


def api_endpoint() -> str:
    """The API endpoint the SDK client talks to.

    make_client passes none, so this is the SDK's own resolution: RUNTA_ENDPOINT,
    else ``endpoint`` in its config file (~/.config/runta/config.toml, or
    RUNTA_CONFIG), else Runta Cloud. It is asked of the SDK rather than read
    off the variable: an endpoint set in the config file would otherwise put
    the runtimes on one deployment and the health probe on another. The SDK
    has no public accessor for it, hence the internal import. The token plays
    no part in the endpoint and a launcher may hold none in its own
    environment, so a placeholder stands in for it.
    """
    from runta._internal.config import resolve_config

    endpoint, _token = resolve_config(endpoint=None, token="unused", config_path=None)
    return endpoint


def ingress() -> tuple[str, str]:
    """``(scheme, domain)`` published ports are served under.

    RUNTA_INGRESS_BASE_URL names both outright; otherwise they follow from the
    API endpoint — Runta Cloud's own domain, or ``<domain>`` for a deployment
    whose API is ``api.<domain>`` — and an endpoint that implies neither is a
    ValueError. No runtime is needed to tell, so it is asked before one exists
    (preflight, create_task_sandbox).
    """
    explicit = os.getenv("RUNTA_INGRESS_BASE_URL", "").strip()
    if explicit:
        parts = urlsplit(explicit)
        return parts.scheme, parts.netloc
    endpoint = api_endpoint()
    api_host = (urlsplit(endpoint).hostname or "").lower()
    domain = _HOSTED_INGRESS_DOMAINS.get(api_host) or api_host.removeprefix("api.")
    if domain == api_host:
        raise ValueError(
            f"cannot tell where {endpoint} serves published ports: "
            "set RUNTA_INGRESS_BASE_URL (e.g. https://ingress.example.com)"
        )
    return "https", domain


def preflight() -> None:
    """The launcher's check of this backend (openenv_sandbox_common.LAUNCH_PREFLIGHTS).

    An endpoint that does not say where published ports are served fails the
    launch here, instead of every episode after its checkpoint build.
    """
    scheme, domain = ingress()
    print(f"Runta published ports: {scheme}://<port>-<runtime id>.{domain}", flush=True)


def base_url(runtime, served_at: tuple[str, str] | None = None) -> str:
    """The env server's externally reachable URL on port 8000.

    A published port is served at ``<scheme>://<port>-<runtime id>.<ingress
    domain>``; *served_at* is ingress()'s answer when the caller already has
    it. The gateway proxies WebSocket as well as HTTP, which OpenEnv needs:
    its client health-checks over HTTP but runs episode I/O over ``/ws``.
    """
    scheme, domain = served_at or ingress()
    return f"{scheme}://{_SERVER_PORT}-{runtime.id}.{domain}"


def _start_keepalive(runtime, task_id: str) -> None:
    """Re-arm the runtime's session limit for as long as THIS process lives.

    A live episode of any length keeps its runtime; a hard-killed caller
    (SIGKILL, OOM, node loss) stops beating and Runta shuts the runtime down
    within _SANDBOX_TTL_S of the last beat. The thread runs until
    delete_task_sandbox stops it, or until re-arms have failed for a whole TTL
    (see _Keepalive and the constants)."""
    keepalive = _Keepalive(
        lambda: runtime.update_timeout_policies(auto_shutdown_after_secs=_SANDBOX_TTL_S),
        f"tb2-sandbox-keepalive-{task_id}",
        interval_s=_KEEPALIVE_INTERVAL_S,
        max_consecutive_failures=_KEEPALIVE_MAX_CONSECUTIVE_FAILURES,
    )
    keepalive.start()
    with _keepalives_lock:
        _keepalives[runtime.id] = keepalive


def _server_run_command(cmd: str) -> str:
    """``docker run`` for the env server, as the container's main process.

    Detached under the Docker daemon rather than backgrounded from the exec: a
    process backgrounded with a plain ``&`` dies when its ``runtime.exec``
    returns. The published port is the one the runtime's ingress serves. The
    entrypoint is overridden because the recipe owns what runs here, whatever
    the task image declares; the image's user and working directory stand.
    """
    return shlex.join(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            _CONTAINER_NAME,
            "--publish",
            f"{_SERVER_PORT}:{_SERVER_PORT}",
            "--entrypoint",
            "bash",
            _IMAGE_TAG,
            "-c",
            cmd,
        ]
    )


def create_task_sandbox(
    task_dir: Path,
    *,
    command_timeout_s: int = recipe.COMMAND_TIMEOUT_S,
    ready_timeout_s: float = _READY_TIMEOUT_S,
):
    """Create ONE per-episode runtime for *task_dir* from its checkpoint.

    Returns ``(runtime, base_url)``. Caller must ``delete_task_sandbox(runtime)``
    when the episode ends — not ``runtime.delete()``, which would race the
    keepalive. First create for a task pays the image build (via
    ``ensure_task_checkpoint``); repeats restore the built checkpoint.

    The checkpoint deliberately holds no running container: the server is
    started here so runtime knobs (TB2_COMMAND_TIMEOUT_S) stay runtime —
    baking them would silently pin them until the next re-bake.
    """
    from runta import IngressSpec

    task_dir = Path(task_dir)
    client = make_client()
    # Asked first: an endpoint that does not say where published ports are
    # served must fail with no checkpoint built and no runtime to delete.
    served_at = ingress()
    checkpoint_id = ensure_task_checkpoint(task_dir)
    # The TTL is deliberately not a parameter: the keepalive thread re-arms
    # _SANDBOX_TTL_S, so a different value passed here would be silently
    # overwritten at the first beat. Nor is the size: a restore comes back at
    # the checkpoint's own (the builder's task_runtime_resources, which the
    # checkpoint's name covers), and the SDK drops vcpus/memory_mib on one.
    runtime = _create_runtime(
        client,
        _runtime_name("run", task_dir),
        checkpoint_id=checkpoint_id,
        ingress_specs=[IngressSpec("https", _SERVER_PORT)],
        auto_shutdown_after_secs=_SANDBOX_TTL_S,
        auto_delete_after_secs=_DELETE_AFTER_S,
    )
    try:
        # Beating from here on, not from when the server answers: the TTL is
        # sized to the gap between beats, and starting the server may take
        # most of a TTL (the exec's own deadline plus ready_timeout_s).
        _start_keepalive(runtime, task_dir.name)
        cmd = server_cmd(command_timeout_s, default_task_id=task_dir.name)
        _checked_exec(runtime, _server_run_command(cmd), timeout=120, what="starting the env server")
        url = base_url(runtime, served_at)
        # wait_server_ready probes with `requests`; the gateway in front of a
        # published port rejects urllib's default user agent (Cloudflare 1010).
        wait_server_ready(url, timeout_s=ready_timeout_s)
        return runtime, url
    except Exception:
        _delete_quietly(runtime)
        raise


def bake(tasks_dir: Path, task_id: str, force: bool) -> None:
    """Pre-build the checkpoint for one task (optional warm cache)."""
    task_dir = tasks_dir / task_id
    name = checkpoint_name(task_dir)
    resources = task_runtime_resources(task_dir)
    print(f"[bake] {name}  vcpus={resources['vcpus']} mem={resources['memory_mib']}MiB")
    checkpoint_id = ensure_task_checkpoint(
        task_dir,
        force=force,
        on_logs=lambda line: print(f"  | {line}", flush=True),
    )
    print(f"[done] {name} ({checkpoint_id})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    recipe.add_task_selection_args(ap)
    ap.add_argument("--force", action="store_true", help="rebuild even when the checkpoint exists")
    args = ap.parse_args()

    tasks_dir, task_ids = recipe.selected_task_ids(args)

    for task_id in task_ids:
        bake(tasks_dir, task_id, args.force)


if __name__ == "__main__":
    sys.exit(main())
