"""Modal materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling
module) and is provider-agnostic. This module is everything Modal-specific
about turning that recipe into a running cloud sandbox:

  ``task_image(...)``   the recipe as a Modal Image: the task's official image
      pulled from its registry, then ONE LAYER PER recipe command, so an edit
      to the recipe re-runs only the layers below it instead of the whole
      chain. Layers are NOT shared between tasks — each task starts from its
      own base image, which makes every hash below it unique — so each task
      pays one cold build of the full chain (measured: ~70 s cold including
      the registry pull, ~9 s warm).
  ``create_task_sandbox(...)``  per-episode create: build-or-cache-hit the
      image, start the env server as the sandbox's entrypoint, expose port
      8000 through an encrypted tunnel, wait for /health, return
      ``(sandbox, base_url)``. The sandbox carries the recipe's ownership tags.

Orphan reclamation differs from the sibling providers by design. A Modal
sandbox's ``timeout`` is a hard ceiling that CANNOT be extended, so the
keepalive-thread dead-man's switch the Daytona and E2B backends use has no
counterpart here — and needs none: ``idle_timeout`` counts an open TCP
connection on a tunnel as activity, and an episode's entire I/O is exactly
that (the OpenEnv client holds a WebSocket open for the episode's lifetime).
A live episode of any length keeps its sandbox; a hard-killed caller (SIGKILL,
OOM, node loss) drops the connection and Modal reclaims the sandbox within
``idle_timeout``. ``timeout`` stays as the ceiling for the one case idleness
cannot catch: an episode that hangs while still connected.

There is deliberately no bake step: layer hashes ARE the cache key here, so
the first create for a task warms exactly what later creates hit, and nothing
a create passes could name a pre-built artifact to use instead.

Sweeping leftovers by hand needs the Python API — Modal's CLI has no
``sandbox`` subcommand. Scoping the listing to the app (``_APP_NAME``) is what
keeps a shared workspace's other workloads out of range, and the recipe's
ownership tags say which task and launcher each sandbox belongs to::

    import asyncio, modal

    async def sweep(kill=False):
        app = await modal.App.lookup.aio("openenv-tbench2", create_if_missing=True)
        async for sb in modal.Sandbox.list.aio(app_id=app.app_id):
            print(sb.object_id, await sb.get_tags.aio())
            if kill:
                await sb.terminate.aio()

    asyncio.run(sweep())

All modal imports are lazy (in-function), mirroring the sibling provider
modules: offline unit tests and non-sandbox launches must not require the SDK.
"""

import os
import threading
from pathlib import Path

import tb2_sandbox_recipe as recipe
from tb2_sandbox_recipe import (
    resolve_docker_image,
    run_with_deadline,
    sandbox_labels,
    server_cmd,
    server_layer_commands,
    task_env_resources,
    wait_server_ready,
)

# Every knob describing ONE Modal sandbox lives here, next to the create that
# uses it; the backend module keeps only the fan-out knobs (how many creates at
# once, how long to keep retrying). All are read at import: a rollout worker is
# a fresh process per run.
#
# The app every per-episode sandbox is created under. Modal requires one, and
# it is what a sweep scopes to — a dedicated app keeps a shared workspace's
# other workloads out of range.
_APP_NAME = os.getenv("OPENENV_MODAL_APP", "openenv-tbench2")

# Lifetime (see the orphan-reclamation note above). The TTL is a HARD ceiling on
# one episode's sandbox: unlike the other backends' TTLs it cannot be extended,
# so it must exceed the longest episode the rollout allows
# (OPENENV_MAX_ROLLOUT_TIME_SECONDS), not merely a heartbeat interval.
_SANDBOX_TTL_S = int(os.getenv("OPENENV_MODAL_SANDBOX_TTL_S", "1800"))
_IDLE_TIMEOUT_S = int(os.getenv("OPENENV_MODAL_IDLE_TIMEOUT_S", "300"))

# How long a create may take (the first one for a task builds the image, which
# Modal does not deadline itself) and how long the env server then has to answer
# /health.
_CREATE_TIMEOUT_S = float(os.getenv("OPENENV_MODAL_CREATE_TIMEOUT_S", "1800"))
_READY_TIMEOUT_S = float(os.getenv("OPENENV_MODAL_READY_TIMEOUT_S", "300"))

# Waiting for the tunnel to be routable is part of the create, not something to
# tune per deployment.
_TUNNEL_TIMEOUT_S = 60.0

# Build logs are worth minutes of silence when bringing up a task by hand.
# Debug-only: enable_output() drives a PROCESS-WIDE output manager, so never
# turn it on with creates fanned out.
_BUILD_LOGS = bool(os.getenv("OPENENV_MODAL_BUILD_LOGS", "").strip())

_ENV_SERVER_PORT = 8000


def task_resources(task_dir: Path) -> dict[str, float | int]:
    """Sandbox size from ``task.toml [environment]``.

    Modal bills per second on ``max(request, actual)``, so the request is the
    task's stated requirement and nothing above it. There is no disk knob to
    map ``storage_mb`` onto — Modal sizes sandbox disk itself.
    """
    cpus, memory_mb, _storage_mb = task_env_resources(task_dir)
    return {"cpu": float(cpus), "memory": memory_mb}


def task_image(task_dir: Path, docker_image: str | None = None):
    """The recipe as a Modal Image, one layer per recipe command.

    Deliberately not one collapsed layer: Modal caches per layer, so editing
    the recipe's tail (a task's own files) or the embedded env source re-runs
    only the layers below the edit rather than reinstalling apt/uv as well.
    That saving is per task, not across the suite — a task's own base image
    makes its whole chain hash differently from every other task's.

    Modal requires linux/amd64 images (all current TB2 task images are), and
    pulls anonymously — a task image behind a private registry would need
    ``from_registry(secret=...)``.
    """
    from modal import Image

    task_dir = Path(task_dir)
    image = Image.from_registry(resolve_docker_image(task_dir, docker_image))
    for command in server_layer_commands(task_dir):
        image = image.run_commands(command)
    return image


_app = None
_app_lock = threading.Lock()


def get_app():
    """The shared App handle, looked up once per process.

    ``App.lookup`` is a network round-trip; a rollout fans out many episodes
    and every create would otherwise repeat it. The first caller pays it under
    the lock, the rest reuse the handle.
    """
    global _app
    with _app_lock:
        if _app is None:
            from modal import App

            _app = App.lookup(_APP_NAME, create_if_missing=True)
        return _app


def base_url(sandbox, *, tunnel_timeout_s: float | None = None) -> str:
    """The env server's externally reachable URL on port 8000.

    Modal's tunnel terminates TLS and forwards at L4, so one host serves both
    halves of the client's traffic: the HTTP health check and the WebSocket
    (/ws) the episode runs over. Tunnel URLs are cryptographically random but
    unauthenticated — whoever holds the URL can reach the server — which is
    another reason the sandbox is per-episode and dies with it. (Modal's
    connect tokens would add auth, but the env client speaks plain HTTP/WS and
    has nowhere to carry the bearer header.)
    """
    timeout = _TUNNEL_TIMEOUT_S if tunnel_timeout_s is None else tunnel_timeout_s
    return sandbox.tunnels(timeout=int(timeout))[_ENV_SERVER_PORT].url


def _exit_detail(sandbox) -> str:
    """Why the server never came up, when the answer is already available.

    The env server IS the sandbox's main process, so a server that failed to
    start leaves an exited sandbox whose output holds the traceback — far more
    useful than a bare ready-timeout. Reading the stream is only safe once it
    has ended (a live sandbox's stdout never closes, and the read would block),
    so this reports nothing for a sandbox that is still running.
    """
    try:
        code = sandbox.poll()
        if code is None:
            return ""
        tail = (sandbox.stdout.read() or "")[-2000:]
        return f"; the sandbox exited with code {code}, server output tail: {tail}"
    except Exception:  # diagnostics must never mask the original failure
        return ""


def create_task_sandbox(
    task_dir: Path,
    *,
    command_timeout_s: int = recipe.COMMAND_TIMEOUT_S,
    ready_timeout_s: float = _READY_TIMEOUT_S,
    create_timeout_s: float = _CREATE_TIMEOUT_S,
    ttl_s: int = _SANDBOX_TTL_S,
    idle_timeout_s: int = _IDLE_TIMEOUT_S,
):
    """Create ONE per-episode sandbox for *task_dir*.

    Returns ``(sandbox, base_url)``. Caller must ``sandbox.terminate()`` when
    the episode ends. The first create for a task pays the image build (Modal
    builds as part of the create); later creates hit the layer cache.

    The env server runs as the sandbox's entrypoint rather than as a follow-up
    exec: Modal would otherwise run the task image's own CMD, and what starts
    in this sandbox must be the recipe's decision, not the task image's. Its
    runtime knobs stay runtime (TB2_COMMAND_TIMEOUT_S is passed per create, not
    baked) so a change takes effect without invalidating the image cache.

    *create_timeout_s* bounds the build+create wall clock, which Modal itself
    does not deadline. The call runs on a scoped thread; on timeout this caller
    stops waiting — freeing its create-throttle slot — and a sandbox that
    materializes anyway is reclaimed by *idle_timeout*, since nothing will ever
    connect to it.
    """
    from modal import Sandbox

    task_dir = Path(task_dir)
    cmd = server_cmd(command_timeout_s, default_task_id=task_dir.name)

    def _create():
        kwargs = dict(
            app=get_app(),
            image=task_image(task_dir),
            encrypted_ports=[_ENV_SERVER_PORT],
            timeout=int(ttl_s),
            idle_timeout=int(idle_timeout_s),
            tags=sandbox_labels(task_dir),
            **task_resources(task_dir),
        )
        if not _BUILD_LOGS:
            return Sandbox.create("bash", "-c", cmd, **kwargs)
        # In-function: only the debug path needs the top-level module, and the
        # offline tests inject a fake `modal` without enable_output.
        import modal

        with modal.enable_output():
            return Sandbox.create("bash", "-c", cmd, **kwargs)

    sandbox = run_with_deadline(_create, create_timeout_s)

    try:
        url = base_url(sandbox)
        wait_server_ready(url, timeout_s=ready_timeout_s)
        return sandbox, url
    except Exception as e:
        detail = _exit_detail(sandbox)
        try:
            sandbox.terminate()
        except Exception:
            pass  # already gone, or the API is down; idle_timeout backstops it
        if detail:
            raise RuntimeError(f"{task_dir.name}: env server did not come up{detail}") from e
        raise
