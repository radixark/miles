"""boxd materialization of the per-task Terminal-Bench-2 sandbox recipe.

The recipe itself — the shell layers that turn a task's official image into a
combined task+env-server image — lives in ``tb2_sandbox_recipe`` (sibling
module) and is provider-agnostic. This module is everything boxd-specific about
turning that recipe into a running sandbox on `boxd <https://boxd.sh/?ref=miles>`_.

A boxd machine is a full Linux VM with its own Docker daemon, so the recipe
runs unchanged as a Dockerfile inside it, and the machine's own HTTPS URL
serves the env server. Its shape is E2B's: a sandbox starts from a **named
snapshot**, built once per task and run ahead of training if you like:

  ``ensure_task_snapshot(...)``  bake the per-task snapshot once, under a
      deterministic name derived from the task id AND a digest of the
      Dockerfile and machine size, so editing the recipe re-bakes
      automatically instead of silently serving a stale snapshot. A builder
      machine sized from ``task.toml`` runs ``docker build`` on the recipe,
      is saved as the snapshot (disk and memory), and is deleted.
  ``create_task_sandbox(...)``  per-episode: a new machine from the snapshot,
      ``docker run`` of the baked image with ``server_cmd()`` publishing port
      8000, wait for /health on the machine's URL, return
      ``(machine_id, base_url)``.
  bake CLI (``python tb2_sandbox_boxd.py ...``)  pre-build snapshots for a
      task list so the first training episode doesn't pay the image build.

Machines are isolated (``isolated=True``): the agent runs model-written
commands, so a sandbox joins no private network and carries no boxd
credentials. Every machine carries the recipe's ownership labels.

Orphans: a caller that dies without reaching its delete (SIGKILL, OOM, node
loss) leaves its machine behind. Two backstops reclaim it. Every machine is
created with a delete-after timer: an episode machine's is the episode cap
(OPENENV_MAX_ROLLOUT_TIME_SECONDS) plus a slack, a builder's covers its build.
Before that, the env client's WebSocket pings keep a live episode's machine
awake, and once the caller is gone the traffic stops and the machine
hibernates after ``OPENENV_BOXD_IDLE_HIBERNATE_S``, which stops its compute.
To delete a run's leftovers sooner, select them by label::

    from boxd import Boxd
    client = Boxd()
    for m in client.machines.list(labels={"openenv-run-id": "<run id>"}):
        client.machines.delete(m.id)

All boxd imports are lazy (in-function), mirroring the sibling provider
modules: offline unit tests and non-sandbox launches must not require the SDK.
"""

import argparse
import hashlib
import os
import re
import secrets
import shlex
import sys
import threading
import time
from pathlib import Path

import tb2_sandbox_recipe as recipe
from tb2_sandbox_recipe import (
    resolve_docker_image,
    sandbox_labels,
    server_cmd,
    server_layer_commands,
    task_env_resources,
    wait_server_ready,
)
from miles.rollout.agentic.credentials import resolve_provider_api_key
from miles.rollout.agentic.rollout_limits import rollout_lock

# Every knob describing ONE boxd machine lives here, next to the create that
# uses it; the backend module keeps only the fan-out knobs. Read at import: a
# rollout worker is a fresh process per run.
_READY_TIMEOUT_S = float(os.getenv("OPENENV_BOXD_READY_TIMEOUT_S", "300"))
_BUILD_TIMEOUT_S = float(os.getenv("OPENENV_BOXD_BUILD_TIMEOUT_S", "1800"))
_IDLE_HIBERNATE_S = int(os.getenv("OPENENV_BOXD_IDLE_HIBERNATE_S", "1800"))
_NAME_PREFIX = os.getenv("OPENENV_BOXD_NAME_PREFIX", "tb2-")
# How far past the episode cap an episode machine's delete-after timer fires:
# the backstop must never end a live episode, only reclaim an orphan.
_MAX_LIFETIME_SLACK_S = 1800

# boxd machine sizes as (vCPU, memory MB), smallest first. A task gets the
# smallest one that covers both its cpus and memory_mb.
_SIZES = ((1, 4096), (2, 8192), (4, 16384), (8, 32768))

_ENV_SERVER_PORT = 8000
_BUILD_DIR = "/home/boxd/openenv-tb2-build"
_CONTAINER = "openenv-server"

# How often to poll a snapshot that is still being saved, and for how long.
_SNAPSHOT_POLL_S = 2.0
_SNAPSHOT_TIMEOUT_S = 600.0

# Machine and snapshot names are DNS labels (the machine URL is
# https://<name>.<domain>), so every name fits in 63 characters.
_MAX_NAME_LEN = 63
_DIGEST_LEN = 10


def machine_size(task_dir: Path) -> tuple[int, int]:
    """``(vcpu, memory_mb)``: the smallest boxd size covering ``task.toml``.

    Disk needs no mapping: every machine has 100 GB, above any task's
    ``storage_mb``. A task asking for more than the largest size is an error
    rather than a silently under-provisioned sandbox.
    """
    cpus, memory_mb, _storage_mb = task_env_resources(task_dir)
    for vcpu, mem in _SIZES:
        if vcpu >= cpus and mem >= memory_mb:
            return vcpu, mem
    raise ValueError(
        f"{task_dir.name}: needs {cpus} CPUs / {memory_mb} MB, above the largest boxd size (8 vCPU / 32 GB)"
    )


def dockerfile(task_dir: Path) -> str:
    """The recipe as a Dockerfile: the official task image plus one RUN per layer.

    Built as root (the TB2 task images are built for a root agent), with the
    image's own entrypoint cleared so ``docker run`` starts the env server
    directly.
    """
    base = resolve_docker_image(task_dir, None)
    lines = [f"FROM {base}", "USER root"]
    lines += [f"RUN {command}" for command in server_layer_commands(task_dir)]
    lines += ["ENTRYPOINT []", 'CMD ["sleep", "infinity"]']
    return "\n".join(lines) + "\n"


def _name(task_dir: Path, suffix: str) -> str:
    """``<prefix><task slug>-<suffix>``, the slug cut so the name fits a DNS label."""
    budget = _MAX_NAME_LEN - len(_NAME_PREFIX) - len(suffix) - 1
    if budget < 1:
        raise ValueError(f"OPENENV_BOXD_NAME_PREFIX {_NAME_PREFIX!r} leaves no room for the task id in a boxd name")
    slug = re.sub(r"[^a-z0-9-]", "-", task_dir.name.lower())[:budget].strip("-")
    return f"{_NAME_PREFIX}{slug}-{suffix}"


def _digest(task_dir: Path) -> str:
    inputs = [dockerfile(task_dir), repr(machine_size(task_dir))]
    return hashlib.sha256("\n".join(inputs).encode()).hexdigest()[:_DIGEST_LEN]


def snapshot_name(task_dir: Path) -> str:
    """Deterministic snapshot name: ``<prefix><task-id>-<recipe digest>``.

    The digest covers the Dockerfile (base image and every build command, which
    embed the tbench2_env source, deterministically tarred) and the machine
    size (a machine from a snapshot keeps the snapshot's size), so the name
    changes exactly when the baked artifact would.
    """
    return _name(Path(task_dir), _digest(Path(task_dir)))


def _image_tag(name: str) -> str:
    return f"openenv-tb2:{name.rsplit('-', 1)[-1]}"


def _machine_name(task_dir: Path) -> str:
    """Unique per machine: names are unique per owner, and one task runs many
    episodes at once. The URL carries no credential, so access to the env
    server rests on this random name and the episode's short lifetime."""
    return _name(task_dir, secrets.token_hex(6))


def _builder_name(task_dir: Path) -> str:
    """Random too: two processes (the bake CLI and a rollout) may bake one
    snapshot at once, and ``rollout_lock`` only orders the episodes of one rollout."""
    return _name(task_dir, f"build-{secrets.token_hex(4)}")


def _max_lifetime_s() -> int:
    # imported on use: the bake CLI runs without the agent loop's dependencies (openai)
    import openenv_agent_function

    return int(openenv_agent_function.MAX_ROLLOUT_TIME_S) + _MAX_LIFETIME_SLACK_S


_DEFAULT_API_KEY_FILE = "~/.config/boxd/api_key"


def resolve_api_key() -> str:
    """The boxd API key: BOXD_API_KEY, else the key file (see
    credentials.resolve_provider_api_key for the file-indirection rationale)."""
    return resolve_provider_api_key("BOXD_API_KEY", "BOXD_API_KEY_FILE", _DEFAULT_API_KEY_FILE)


def _org() -> str | None:
    """BOXD_ORG, when set: the org machines and snapshots belong to. Unset, the
    key's own org is used (an API key is issued for exactly one org)."""
    return os.getenv("BOXD_ORG", "").strip() or None


_client_lock = threading.Lock()
_client = None


def make_client():
    """The process's boxd client: key from resolve_api_key(), region from
    optional BOXD_REGION (read by the SDK). Built once and shared, because this
    runs once per create attempt, retries included."""
    global _client
    with _client_lock:
        if _client is None:
            from boxd import Boxd

            _client = Boxd(api_key=resolve_api_key())
        return _client


def _exec(client, machine_id: str, command: str, *, timeout: float) -> str:
    result = client.machines.exec(machine_id, command, timeout=timeout)
    if result.exit_code != 0:
        output = (result.stdout + result.stderr)[-4000:]
        raise RuntimeError(f"`{command[:80]}` exited {result.exit_code} on boxd machine {machine_id}:\n{output}")
    return result.stdout


def _snapshot_status(client, name: str) -> str | None:
    """The snapshot's status ("pending", "ready", "failed"), or None when absent."""
    from boxd import NotFoundError

    try:
        snapshot = client.snapshots.get(name, org=_org())
    except NotFoundError:
        return None
    if snapshot.archived:
        # machines can't be created from an archived snapshot, nor can it be saved again
        raise RuntimeError(f"boxd snapshot {name} is archived ({snapshot.storage}); unarchive it to use it")
    return snapshot.status


def _wait_snapshot_ready(client, name: str, timeout_s: float = _SNAPSHOT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while (status := _snapshot_status(client, name)) == "pending":
        if time.monotonic() > deadline:
            raise TimeoutError(f"boxd snapshot {name} not ready in {timeout_s}s")
        time.sleep(_SNAPSHOT_POLL_S)
    if status != "ready":
        raise RuntimeError(f"boxd snapshot {name} is {status!r}, not ready")


def _snapshot_labels(task_dir: Path) -> dict[str, str]:
    """The builder's labels, which its snapshot hands to every machine made from
    it. No run id: one snapshot serves many runs, and a restore merges the
    episode's own labels over these."""
    labels = sandbox_labels(task_dir)
    labels.pop("openenv-run-id", None)
    return labels


def _bake(client, task_dir: Path, name: str, *, build_timeout_s: float, on_logs=None) -> None:
    """Build the task image on a fresh builder machine, then save it as *name*."""
    vcpu, memory_mb = machine_size(task_dir)
    builder = client.machines.create(
        _builder_name(task_dir),
        org=_org(),
        isolated=True,
        labels=_snapshot_labels(task_dir),
        vcpu=vcpu,
        memory=f"{memory_mb // 1024}G",
        auto_suspend_timeout=0,
        delete_after=int(build_timeout_s + _READY_TIMEOUT_S + _SNAPSHOT_TIMEOUT_S) + _MAX_LIFETIME_SLACK_S,
    )
    try:
        client.machines.wait_until_ready(builder.id, timeout=_READY_TIMEOUT_S)
        client.machines.files.upload(builder.id, f"{_BUILD_DIR}/Dockerfile", dockerfile(task_dir))
        log = _exec(
            client,
            builder.id,
            f"docker build --progress=plain -t {_image_tag(name)} {_BUILD_DIR} 2>&1",
            timeout=build_timeout_s,
        )
        if on_logs:
            for line in log.splitlines():
                on_logs(line)
        client.snapshots.create(builder.id, name)
        _wait_snapshot_ready(client, name)
    finally:
        try:
            client.machines.delete(builder.id)
        except Exception:
            pass  # cleanup must not mask the real failure; its delete-after timer reclaims it


def ensure_task_snapshot(
    task_dir: Path,
    *,
    force: bool = False,
    build_timeout_s: float = _BUILD_TIMEOUT_S,
    on_logs=None,
) -> str:
    """Bake the per-task snapshot if its name doesn't exist yet; return the name."""
    task_dir = Path(task_dir)
    client = make_client()
    name = snapshot_name(task_dir)
    # one build per name per rollout: a rollout fans out many episodes of the same task at once,
    # and every concurrent miss would otherwise start its own multi-minute build
    with rollout_lock(f"openenv-boxd-build-{name}"):
        status = None if force else _snapshot_status(client, name)
        if status == "pending":
            _wait_snapshot_ready(client, name)
        elif status != "ready":
            _bake(client, task_dir, name, build_timeout_s=build_timeout_s, on_logs=on_logs)
    return name


def _start_server_command(name: str, command_timeout_s: int, task_id: str) -> str:
    cmd = server_cmd(command_timeout_s, default_task_id=task_id)
    return (
        f"docker run -d --name {_CONTAINER} -p {_ENV_SERVER_PORT}:{_ENV_SERVER_PORT} "
        f"{_image_tag(name)} bash -c {shlex.quote(cmd)}"
    )


def create_task_sandbox(
    task_dir: Path,
    *,
    command_timeout_s: int = recipe.COMMAND_TIMEOUT_S,
    ready_timeout_s: float = _READY_TIMEOUT_S,
):
    """Create ONE per-episode machine for *task_dir* from its snapshot.

    Returns ``(machine_id, base_url)``. Caller must ``delete_sandbox(machine_id)``
    when the episode ends. First create for a task pays the snapshot bake (via
    ``ensure_task_snapshot``); repeats start from the snapshot.

    The snapshot holds the built image, not a running server: the server is
    started here so runtime knobs (TB2_COMMAND_TIMEOUT_S) stay runtime.
    """
    task_dir = Path(task_dir)
    client = make_client()
    name = ensure_task_snapshot(task_dir)
    machine = client.machines.create(
        _machine_name(task_dir),
        from_snapshot=name,
        org=_org(),
        isolated=True,
        # merged over the snapshot's labels (see _snapshot_labels)
        labels=sandbox_labels(task_dir),
        auto_suspend_timeout=0,
        delete_after=_max_lifetime_s(),
    )
    try:
        machine = client.machines.wait_until_ready(machine.id, timeout=ready_timeout_s)
        # The env client is a program, not a browser: no browser check on the URL.
        client.machines.set_bot_protection(machine.id, mode="off")
        client.machines.set_auto_hibernate_timeout(machine.id, _IDLE_HIBERNATE_S)
        _exec(client, machine.id, _start_server_command(name, command_timeout_s, task_dir.name), timeout=60)
        url = machine.access.url
        wait_server_ready(url, timeout_s=ready_timeout_s)
        return machine.id, url
    except Exception:
        try:
            client.machines.delete(machine.id)
        except Exception:
            pass  # cleanup must not mask the real failure; its delete-after timer reclaims it
        raise


def delete_sandbox(machine_id: str) -> None:
    """Delete a machine created here."""
    make_client().machines.delete(machine_id)


def bake(tasks_dir: Path, task_id: str, force: bool) -> None:
    """Pre-build the snapshot for one task (optional warm cache)."""
    task_dir = tasks_dir / task_id
    name = snapshot_name(task_dir)
    vcpu, memory_mb = machine_size(task_dir)
    print(f"[bake] {name}  vcpu={vcpu} mem={memory_mb}MB")
    ensure_task_snapshot(task_dir, force=force, on_logs=lambda line: print(f"  | {line}", flush=True))
    print(f"[done] {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    recipe.add_task_selection_args(ap)
    ap.add_argument("--force", action="store_true", help="rebuild even when the snapshot exists")
    args = ap.parse_args()

    tasks_dir, task_ids = recipe.selected_task_ids(args)

    for task_id in task_ids:
        bake(tasks_dir, task_id, args.force)


if __name__ == "__main__":
    sys.exit(main())
