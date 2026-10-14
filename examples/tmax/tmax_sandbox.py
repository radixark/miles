"""Per-episode TMax sandbox on a local Podman (or Docker) daemon.

A port of the open-instruct ``swerl_vanillux_sandbox`` environment and its
``DockerBackend`` onto the Docker-compatible REST API that ``podman system
service`` exposes. Each episode gets a fresh container from the task's
prebuilt image; the agent's only tool is bash; tests are uploaded only at
submit time so the policy cannot read them.

Behavior kept identical to open-instruct, since it shapes the reward:

* the persistent-shell wrapper (cwd + exported env survive between execs),
* mini-swe-agent head/tail observation truncation plus ``(exit_code=N)``,
* the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` submit marker,
* reward = ``/logs/verifier/reward.txt`` clamped to [0, 1], 0 when missing,
* an OOM-killed sandbox ends the episode with reward 0,
* a failed start is retried 5 times with 1-16 s backoff, and a daemon that fails with a connectivity
  error is benched for 300-330 s (open-instruct's SWERLVanilluxSandboxEnv.reset and EnvironmentPool),
* a container that vanished or stopped (not OOM) mid-episode is recreated once and the command retried,
  losing its state, as DockerBackend._restart_and_retry_exec does.

Under ``--custom-agent-function-mode subproc`` (the default) each episode runs in
a process of its own, so the limits shared across episodes (start and exec
concurrency, one pull per image) are ``rollout_limits`` file locks scoped to the
rollout process, and containers are labelled with that scope rather than a pid.

Env vars:
  TMAX_DOCKER_HOST          Docker API endpoint(s), comma-separated (default: $DOCKER_HOST,
                            else unix:///run/podman/podman.sock). One podman daemon serializes
                            on its lock database at high concurrency, so open-instruct runs
                            several per node; each episode picks one at random and keeps it.
  TMAX_SANDBOX_MEM_LIMIT    per-container memory limit (default: 4g).
  TMAX_COMMAND_TIMEOUT_S    per-bash-command timeout (default: 120).
  TMAX_TEST_TIMEOUT_S       verifier timeout (default: 600, open-instruct's floor).
  TMAX_START_CONCURRENCY    concurrent container starts per rollout process (default: 16).
  TMAX_EXEC_CONCURRENCY     concurrent execs per rollout process (default: 128).
  TMAX_REGISTRY_USERNAME /  optional pull credentials. Always sent explicitly, so the
  TMAX_REGISTRY_PASSWORD    client's ~/.docker/config.json credential helper is never
                            consulted; without them Podman uses its own auth.json.
"""

import asyncio
import io
import logging
import os
import random
import shlex
import socket
import tarfile
import tempfile
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import docker

from miles.rollout.agentic.rollout_limits import current_scope, rollout_semaphore

logger = logging.getLogger(__name__)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SANDBOX_LABEL = "miles_tmax_sandbox"
_OWNER_LABEL = f"{SANDBOX_LABEL}_owner"

_MAX_OUTPUT_BYTES = 1_000_000
_OBS_MAX_CHARS = 10_000
_OBS_HEAD_CHARS = 5_000
_OBS_TAIL_CHARS = 5_000
_OBS_TOO_LONG_HINT = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a\n"
    "smaller number of lines selectively. If you're using grep or find and it\n"
    "produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you\n"
    "can redirect output to a file and then search in that file.\n"
)
_TRANSIENT_EXEC_MARKERS = ("database is locked", "retrieving exec session", "timed out waiting for file")
_TRANSIENT_EXEC_RETRIES = 5
# open-instruct backends.DOCKER_HOST_CONNECTIVITY_ERROR_MARKERS
_CONNECTIVITY_MARKERS = (
    "error while fetching server api version",
    "unixhttpconnectionpool",
    "read timed out",
    "connection refused",
    "connection aborted",
    "broken pipe",
    "connection reset",
)
_START_ATTEMPTS = 5
_START_RETRY_BASE_S = 1.0
_START_RETRY_MAX_S = 16.0
_START_RETRY_JITTER_S = 2.0
_HOST_COOLDOWN_S = 300.0
_HOST_COOLDOWN_JITTER_S = 30.0

_BASH_WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
_BASH_CWD_PATH = "/tmp/.swerl_vanillux_cwd"
_BASH_ENV_PATH = "/tmp/.swerl_vanillux_env"
_BASH_WRAPPER = f"""#!/bin/bash
set -a
source {shlex.quote(_BASH_ENV_PATH)} 2>/dev/null || true
set +a
_cwd="$(cat {shlex.quote(_BASH_CWD_PATH)} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {shlex.quote(_BASH_ENV_PATH)}
pwd > {shlex.quote(_BASH_CWD_PATH)}
exit $_exit_code
"""


class SandboxInfraError(RuntimeError):
    """The sandbox could not be provided; not the policy's fault."""


class SandboxOOMError(RuntimeError):
    """The sandbox was OOM-killed; the episode ends with reward 0."""


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class _Settings:
    docker_hosts: tuple[str, ...]
    mem_limit: str
    command_timeout_s: int
    test_timeout_s: int
    start_concurrency: int
    exec_concurrency: int
    registry_auth: dict[str, str]


@cache
def _settings() -> _Settings:
    return _Settings(
        docker_hosts=tuple(
            host.strip()
            for host in (
                os.getenv("TMAX_DOCKER_HOST") or os.getenv("DOCKER_HOST") or "unix:///run/podman/podman.sock"
            ).split(",")
            if host.strip()
        ),
        mem_limit=os.getenv("TMAX_SANDBOX_MEM_LIMIT", "4g"),
        command_timeout_s=int(os.getenv("TMAX_COMMAND_TIMEOUT_S", "120")),
        test_timeout_s=int(os.getenv("TMAX_TEST_TIMEOUT_S", "600")),
        start_concurrency=int(os.getenv("TMAX_START_CONCURRENCY", "16")),
        exec_concurrency=int(os.getenv("TMAX_EXEC_CONCURRENCY", "128")),
        registry_auth=(
            {"username": os.environ["TMAX_REGISTRY_USERNAME"], "password": os.environ["TMAX_REGISTRY_PASSWORD"]}
            if os.getenv("TMAX_REGISTRY_USERNAME")
            else {}
        ),
    )


@cache
def _client(docker_host: str) -> docker.DockerClient:
    return docker.DockerClient(base_url=docker_host, timeout=300)


def _bench_dir() -> Path:
    bench_dir = Path(tempfile.gettempdir()) / f"miles-tmax-benched-hosts-{current_scope()}"
    bench_dir.mkdir(exist_ok=True)
    return bench_dir


def _bench_path(docker_host: str) -> Path:
    return _bench_dir() / "".join(c if c.isalnum() else "_" for c in docker_host)


def _bench(docker_host: str, error: BaseException) -> None:
    """Keep every episode of this rollout off a daemon that just failed to connect."""
    cooldown = _HOST_COOLDOWN_S + random.uniform(0.0, _HOST_COOLDOWN_JITTER_S)
    _bench_path(docker_host).write_text(str(time.time() + cooldown))
    logger.warning(f"benching sandbox daemon {docker_host} for {cooldown:.0f}s after: {error!r}")


def _pick_host() -> str:
    """A random daemon that is not benched; any daemon if all of them are."""
    hosts = _settings().docker_hosts
    now = time.time()

    def benched(host: str) -> bool:
        try:
            return float(_bench_path(host).read_text()) > now
        except (FileNotFoundError, ValueError):
            return False

    return random.choice([host for host in hosts if not benched(host)] or list(hosts))


def owner() -> str:
    """The rollout process this process's sandboxes belong to; shared by its agent processes."""
    return f"{socket.gethostname()}-{current_scope()}"


def truncate_observation(output: str) -> str:
    """mini-swe-agent head/tail truncation, as in open-instruct's vanillux env."""
    if len(output) <= _OBS_MAX_CHARS:
        return output
    elided = len(output) - _OBS_HEAD_CHARS - _OBS_TAIL_CHARS
    return (
        f"{_OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({_OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:_OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({_OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-_OBS_TAIL_CHARS:]}"
    )


def _tar_directory(host_dir: str, container_dir: str) -> bytes:
    prefix = container_dir.lstrip("/")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(host_dir, arcname=prefix)
    return buf.getvalue()


def _tar_file(path: str, content: str, mode: int = 0o755) -> bytes:
    data = content.encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=path.lstrip("/"))
        info.size = len(data)
        info.mode = mode
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TmaxSandbox:
    """One task container for one episode. Use as ``async with TmaxSandbox(...) as box``."""

    def __init__(self, image: str, task_dir: str):
        self._image = image
        self._task_dir = task_dir
        self._tests_dir = os.path.join(task_dir, "tests")
        self._docker_host = _pick_host()
        self._container = None
        self._ready = False  # set once the episode's container is up; a restart is only allowed after that
        self._restarted = False

    async def __aenter__(self) -> "TmaxSandbox":
        last_error: Exception | None = None
        for attempt in range(_START_ATTEMPTS):
            try:
                await self._start()
                await self._prepare_runtime()
                self._ready = True
                return self
            except asyncio.CancelledError:
                await asyncio.shield(self.close())
                raise
            except Exception as e:
                last_error = e
                await self.close()
                if any(marker in str(e).lower() for marker in _CONNECTIVITY_MARKERS):
                    _bench(self._docker_host, e)
                if attempt + 1 < _START_ATTEMPTS:
                    delay = min(_START_RETRY_BASE_S * 2**attempt, _START_RETRY_MAX_S)
                    await asyncio.sleep(delay + random.uniform(0.0, _START_RETRY_JITTER_S))
                    self._docker_host = _pick_host()
        raise SandboxInfraError(
            f"sandbox start failed {_START_ATTEMPTS}x for image={self._image}: {last_error!r}"
        ) from last_error

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def bash(self, command: str) -> tuple[str, bool]:
        """Run one agent command. Returns (observation, submitted)."""
        result = await self._exec(f"bash {shlex.quote(_BASH_WRAPPER_PATH)} {shlex.quote(command)}")
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        if SUBMIT_MARKER in output:
            return output, True
        observation = truncate_observation(output) if output else "(no output)"
        return f"{observation}\n\n(exit_code={result.exit_code})", False

    async def run_tests(self) -> float:
        """Upload the withheld tests, run them, and return the clamped reward."""
        if not os.path.isfile(os.path.join(self._tests_dir, "test.sh")):
            raise SandboxInfraError(f"no tests/test.sh under {self._task_dir}")
        try:
            await self._put_archive(_tar_directory(self._tests_dir, "/tests"))
            await self._exec("chmod +x /tests/test.sh 2>/dev/null || true")
        except SandboxOOMError:
            raise
        except Exception as e:
            raise SandboxInfraError(f"tests upload failed: {e!r}") from e
        await self._exec("bash /tests/test.sh", timeout_s=max(_settings().test_timeout_s, 600))
        reward_result = await self._exec("cat /logs/verifier/reward.txt")
        try:
            return max(0.0, min(1.0, float(reward_result.stdout.strip())))
        except ValueError:
            return 0.0

    async def close(self) -> None:
        container, self._container = self._container, None
        if container is not None:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                logger.warning(f"failed to remove sandbox {container.short_id}: {e!r}")

    async def _start(self) -> None:
        async with rollout_semaphore("tmax-start", _settings().start_concurrency):
            await self._ensure_image()
            self._container = await asyncio.to_thread(
                _client(self._docker_host).containers.create,
                self._image,
                command="sleep infinity",
                detach=True,
                labels={SANDBOX_LABEL: "1", _OWNER_LABEL: owner()},
                mem_limit=_settings().mem_limit,
                memswap_limit=_settings().mem_limit,
            )
            await asyncio.to_thread(self._container.start)

    async def _ensure_image(self) -> None:
        # Siblings of a GRPO group start the same image at once; pull it once per daemon (each has its own store).
        client = _client(self._docker_host)
        async with rollout_semaphore(f"tmax-pull-{self._docker_host}-{self._image}", 1):
            try:
                await asyncio.to_thread(client.images.get, self._image)
            except docker.errors.ImageNotFound:
                logger.info(f"pulling sandbox image {self._image} on {self._docker_host}")
                await asyncio.to_thread(client.images.pull, self._image, auth_config=_settings().registry_auth)

    async def _prepare_runtime(self) -> None:
        await self._exec("mkdir -p /workspace /output /logs/verifier /root")
        seeds_dir = os.path.join(self._task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            await self._put_archive(_tar_directory(seeds_dir, "/workspace"))
        await self._exec(
            "cd /workspace && "
            '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
            f"printf '%s\\n' /app > {shlex.quote(_BASH_CWD_PATH)} && "
            f": > {shlex.quote(_BASH_ENV_PATH)}"
        )
        await self._put_archive(_tar_file(_BASH_WRAPPER_PATH, _BASH_WRAPPER))

    async def _put_archive(self, tar_bytes: bytes) -> None:
        ok = await asyncio.to_thread(self._container.put_archive, "/", tar_bytes)
        if not ok:
            raise SandboxInfraError("put_archive returned False")

    async def _exec(self, command: str, timeout_s: int | None = None) -> ExecResult:
        timeout_s = timeout_s or _settings().command_timeout_s
        wrapped = f"timeout --signal=TERM --kill-after=10 {timeout_s} bash -c {shlex.quote(command)}"
        async with rollout_semaphore("tmax-exec", _settings().exec_concurrency):
            exit_code, output = await self._exec_with_retries(wrapped)
        stdout = ((output[0] or b"") if output else b"")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = ((output[1] or b"") if output else b"")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if exit_code == 124:
            stderr = f"Command timed out after {timeout_s}s.\n{stderr}"
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def _exec_with_retries(self, wrapped: str):
        for attempt in range(_TRANSIENT_EXEC_RETRIES):
            try:
                return await asyncio.to_thread(self._container.exec_run, ["bash", "-c", wrapped], demux=True)
            except docker.errors.APIError as e:
                gone = isinstance(e, docker.errors.NotFound) or getattr(e, "status_code", None) == 409
                if getattr(e, "status_code", None) == 409 and await self._was_oom_killed():
                    raise SandboxOOMError(f"sandbox {self._container.short_id} was OOM-killed") from e
                if gone and self._ready and not self._restarted:
                    await self._restart(e)
                    continue
                transient = any(marker in str(e).lower() for marker in _TRANSIENT_EXEC_MARKERS)
                if not transient or attempt + 1 == _TRANSIENT_EXEC_RETRIES:
                    raise SandboxInfraError(f"exec failed: {e!r}") from e
                await asyncio.sleep(min(0.5 * 2**attempt, 8.0))

    async def _restart(self, error: BaseException) -> None:
        logger.warning(f"sandbox for {self._image} is gone ({error!r}); recreating it once, state is lost")
        self._restarted = True
        await self.close()
        try:
            await self._start()
            await self._prepare_runtime()
        except Exception as e:
            raise SandboxInfraError(f"sandbox restart failed: {e!r}") from e

    async def _was_oom_killed(self) -> bool:
        try:
            await asyncio.to_thread(self._container.reload)
        except Exception:
            return False
        return bool(self._container.attrs.get("State", {}).get("OOMKilled"))


async def remove_owned_sandboxes() -> int:
    """Force-remove every sandbox this rollout process's episodes created, on every daemon. Returns the count."""
    removed = 0
    for docker_host in _settings().docker_hosts:
        try:
            containers = await asyncio.to_thread(
                lambda host=docker_host: _client(host).containers.list(
                    all=True, filters={"label": f"{_OWNER_LABEL}={owner()}"}
                )
            )
        except docker.errors.DockerException as e:  # one unreachable daemon must not stop the sweep of the others
            logger.warning(f"cannot list sandboxes on {docker_host}: {e!r}")
            continue
        for container in containers:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except docker.errors.NotFound:
                pass
        removed += len(containers)
    return removed
