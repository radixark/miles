"""The gateway sandbox backend: TMax containers on a LiteRegistry podman fleet.

The fleet is a separate deployment (https://github.com/goncalorafaria/literegistry, ``literegistry_podman_beaker``):
rootless podman replicas and Docker Hub mirrors that heartbeat into a Redis registry, behind a gateway that routes each
container's requests to the replica that owns it. The training job runs no podman, so no sandbox shares a node with
Megatron or SGLang. This follows open-instruct's ``GatewayBackend`` (branch ``literegistry_podman``) and talks to the
gateway through literegistry's own client, ``literegistry-podman-client``, which owns the API and its error types; it
tracks the current literegistry release, not older ones.

Per episode, through a ``PodmanSession``:

* handshake: the gateway picks a replica, which starts a container from the task image (pulled through the fleet's
  mirrors) and binds its affinity id to itself;
* execute: one ``podman exec`` on that replica. An exec is capped at 60 s, 1 MB of stdin and 16 KB of command, so a
  command never runs as a bare exec: one exec launches it detached with its output in files, then wait execs block
  in the container until it finishes and return its exit code and bounded output;
* close: removes the container.

The gateway sends a pinned command once and reports a transport failure (``502``/``503``/``504``) instead of
replaying it; every request here is safe to repeat (a ``mkdir`` lock makes a repeated launch a no-op, each upload
chunk is its own file), so those are retried. The binding expires after 15 idle minutes, so a keepalive exec runs
while the container idles through a long generation.

A container the fleet reports lost (``PodmanContainerLostError``: the replica's watchdog removed it past its limits,
4 GB of RSS or 2048 processes on 2026-10-16, it stopped because its init process died, or its replica left the
roster) ends the episode (``SandboxLostError``, end reason ``sandbox_lost``, its reason in the trace). So does a
binding the gateway no longer has (``404``; the keepalive prevents expiry): nothing can reach the container again.
tmax_agent_function scores the policy's losses 0 and drops the group for the fleet's. As literegistry prescribes,
the episode does not continue in a replacement container.
"""

import asyncio
import base64
import logging
import random
import shlex
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from literegistry_podman_client import (
    CommandResult,
    PodmanClient,
    PodmanContainerLostError,
    PodmanGatewayError,
    PodmanSession,
)
from tmax_sandbox import (
    _START_ATTEMPTS,
    _START_RETRY_BASE_S,
    _START_RETRY_JITTER_S,
    _START_RETRY_MAX_S,
    ExecResult,
    SandboxBase,
    SandboxInfraError,
    SandboxLostError,
    _settings,
)

from miles.rollout.agentic.rollout_limits import current_scope, rollout_semaphore

logger = logging.getLogger(__name__)

# The replica's per-exec contract: timeout <= 60 s, stdin <= 1 MB, command <= 16 KB.
_EXEC_TIMEOUT_CAP_S = 59.0
# Base64 inflates by 4/3; stay well below the 1 MB stdin cap.
_UPLOAD_CHUNK_BYTES = 512 * 1024
# Well below the replica's output caps.
_MAX_STDOUT_BYTES = 512 * 1024
_MAX_STDERR_BYTES = 128 * 1024
# A wait exec blocks in the container until the command finishes or this budget runs out, so a finished command
# costs one round trip and a long one a request per ~30 s; the gap to the exec timeout absorbs `podman exec`
# spawn latency on a loaded replica (open-instruct saw more than 20 s).
_WAIT_EXEC_TIMEOUT_S = 55.0
_IN_CONTAINER_WAIT_S = 30
# A wait exec that returns neither marker was cut short itself; the command's state lives in files, so it repeats.
_MAX_MARKERLESS_WAITS = 3
_DONE_MARKER = "__SWERL_GATEWAY_DONE__"
_PENDING_MARKER = "__SWERL_GATEWAY_PENDING__"
_WORK_DIR = "/tmp/.swerl_gateway"

_HANDSHAKE_ATTEMPTS = 5
_REQUEST_ATTEMPTS = 4
_REQUEST_TIMEOUT_S = 300.0  # a handshake pulls the image; an exec returns within its own timeout
_RETRY_BASE_S = 0.5
_RETRY_MAX_S = 8.0
# Transient gateway answers: a replica still starting the exec, an owner briefly unreachable, a dropped connection.
# A request that failed without a status never reached the gateway.
_TRANSIENT_STATUSES = frozenset({408, 500, 502, 503, 504})


def _live_dir() -> Path:
    """This rollout's live containers, one empty file per affinity id, for the abort hook."""
    path = Path(tempfile.gettempdir()) / f"miles-tmax-gateway-live-{current_scope()}"
    path.mkdir(exist_ok=True)
    return path


def _client() -> PodmanClient:
    return PodmanClient(_settings().gateway_url, workdir="/", request_timeout=_REQUEST_TIMEOUT_S)


class TmaxGatewaySandbox(SandboxBase):
    """The gateway backend: a container on a LiteRegistry podman replica."""

    def __init__(self, image: str, task_dir: str):
        super().__init__(image, task_dir)
        if not _settings().gateway_url:  # a configuration error, not an infra failure to score as 0
            raise ValueError("the gateway sandbox backend needs TMAX_GATEWAY_URL")
        self._client: PodmanClient | None = None
        self._session: PodmanSession | None = None
        self._lock = asyncio.Lock()  # one exec at a time per container; the keepalive skips while one runs
        self._last_request = time.monotonic()
        self._keepalive: asyncio.Task | None = None

    async def __aenter__(self) -> "TmaxGatewaySandbox":
        self._client = await _client().open()
        last_error: Exception | None = None
        for attempt in range(_START_ATTEMPTS):
            try:
                await self._start()
                await self._prepare_runtime()
                if _settings().gateway_keepalive_s > 0:
                    self._keepalive = asyncio.create_task(self._keepalive_loop())
                return self
            except asyncio.CancelledError:
                await asyncio.shield(self._release())
                raise
            except Exception as e:  # a lost or failed container before the episode started: try a fresh one
                last_error = e
                await self._close_container()
                if attempt + 1 < _START_ATTEMPTS:
                    delay = min(_START_RETRY_BASE_S * 2**attempt, _START_RETRY_MAX_S)
                    await asyncio.sleep(delay + random.uniform(0.0, _START_RETRY_JITTER_S))
        await self._release()
        raise SandboxInfraError(
            f"sandbox start failed {_START_ATTEMPTS}x for image={self._image}: {last_error!r}"
        ) from last_error

    async def _release(self) -> None:
        if self._keepalive is not None:
            self._keepalive.cancel()
            self._keepalive = None
        try:
            await self._close_container()
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    @property
    def instance_id(self) -> str | None:
        """The replica running the container."""
        return self._session.instance_id if self._session is not None else None

    # ------------------------------------------------------------------ container lifecycle

    async def _start(self) -> None:
        client, client_id = self._client, f"miles-tmax-{uuid.uuid4().hex[:12]}"
        async with rollout_semaphore("tmax-gateway-start", _settings().start_concurrency):
            self._session = await _retry(
                lambda: client.handshake(image=self._image, client_id=client_id), attempts=_HANDSHAKE_ATTEMPTS
            )
        (_live_dir() / self._session.affinity_id).touch()

    async def _close_container(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        (_live_dir() / session.affinity_id).unlink(missing_ok=True)
        try:
            await session.close()  # a container already lost counts as closed
        except PodmanGatewayError as e:  # the replica's janitor removes it later; never fail the episode on it
            logger.warning(f"failed to close gateway sandbox {session.affinity_id[:12]}: {e}")

    # ------------------------------------------------------------------ commands

    async def _exec(self, command: str, timeout_s: int | None = None) -> ExecResult:
        try:
            return await self._run_command(command, timeout_s or _settings().command_timeout_s)
        except PodmanContainerLostError as e:
            raise SandboxLostError(f"sandbox {self._image} lost: {_describe_loss(e)}", reason=e.reason) from e
        except PodmanGatewayError as e:
            if e.status_code == 404:  # the binding is gone, so is any way back to the container
                raise SandboxLostError(f"sandbox {self._image} unreachable: {e}", reason="binding_lost") from e
            raise SandboxInfraError(f"gateway exec failed: {e}") from e

    async def _run_command(self, command: str, timeout_s: float) -> ExecResult:
        """Launch ``command`` detached in the container, then wait for it and read its bounded output."""
        job_dir = f"{_WORK_DIR}/{uuid.uuid4().hex[:16]}"
        quoted_dir = shlex.quote(job_dir)
        detached = shlex.quote(
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(timeout_s))} "
            f"bash {job_dir}/cmd.sh > {job_dir}/out 2> {job_dir}/err < /dev/null; "
            f"echo $? > {job_dir}/rc.tmp && mv {job_dir}/rc.tmp {job_dir}/rc"
        )
        # The job dir is the lock: a repeated launch finds it and only reports success. `{ ... & }` backgrounds just
        # the setsid command, so `cat` still reads the request's stdin, which carries the command.
        launcher = (
            f"mkdir -p {shlex.quote(_WORK_DIR)} && "
            f"if mkdir {quoted_dir} 2> /dev/null; then "
            f"cat > {quoted_dir}/cmd.sh && "
            "{ setsid nohup bash -c " + detached + " > /dev/null 2>&1 < /dev/null & } && echo LAUNCHED; "
            "else echo LAUNCHED; fi"
        )
        waiter = (
            f"if timeout {_IN_CONTAINER_WAIT_S} bash -c 'until [ -f {job_dir}/rc ]; do sleep 0.1; done'; then "
            f'echo {_DONE_MARKER}; echo "RC=$(cat {quoted_dir}/rc)"; '
            f"head -c {_MAX_STDOUT_BYTES} {quoted_dir}/out; "
            f"head -c {_MAX_STDERR_BYTES} {quoted_dir}/err >&2; "
            f"else echo {_PENDING_MARKER}; fi"
        )
        async with self._lock:
            launch = await self._call(launcher, stdin=command, timeout=30.0)
            if "LAUNCHED" not in launch.stdout:
                raise PodmanGatewayError(f"command launcher failed (exit={launch.exit_code}): {launch.stderr[:500]}")
            # The command's own timeout, plus TERM -> KILL grace and slack for the launcher.
            deadline = time.monotonic() + timeout_s + 60.0
            markerless = 0
            while True:
                response = await self._call(waiter, timeout=_WAIT_EXEC_TIMEOUT_S)
                if _DONE_MARKER in response.stdout:
                    break
                if _PENDING_MARKER not in response.stdout:
                    markerless += 1
                    if markerless >= _MAX_MARKERLESS_WAITS:
                        # A container being torn down, or a very slow replica: a trivial exec tells them apart
                        # (a lost container raises here) before this becomes a plain tool error.
                        await self._call("true", timeout=15.0)
                        raise PodmanGatewayError(
                            f"wait exec returned no marker {markerless}x (exit={response.exit_code}): "
                            f"stdout={response.stdout[:200]!r} stderr={response.stderr[:200]!r}"
                        )
                if time.monotonic() > deadline:
                    await self._cleanup(quoted_dir)
                    raise PodmanGatewayError(f"command did not finish within {timeout_s}s plus grace")
            await self._cleanup(quoted_dir)
        rc_line, _, out = response.stdout.split(_DONE_MARKER, 1)[1].lstrip("\n").partition("\n")
        try:
            exit_code = int(rc_line.removeprefix("RC=").strip())
        except ValueError:
            raise PodmanGatewayError(f"wait exec returned an invalid rc line: {rc_line!r}") from None
        stderr = response.stderr
        if exit_code == 124:
            stderr = f"Command timed out after {timeout_s}s.\n{stderr}"
        return ExecResult(stdout=out, stderr=stderr, exit_code=exit_code)

    async def _call(self, command: str, stdin: str = "", timeout: float = 30.0) -> CommandResult:
        """One bounded ``podman exec`` on the container's replica, retried while the gateway answers transiently."""
        session = self._session
        if session is None:
            raise PodmanGatewayError("no container")
        self._last_request = time.monotonic()
        return await _retry(
            lambda: session.execute(command, stdin=stdin, timeout=min(timeout, _EXEC_TIMEOUT_CAP_S)),
            attempts=_REQUEST_ATTEMPTS,
        )

    async def _cleanup(self, quoted_dir: str) -> None:
        """Remove a finished command's scratch dir; the result is in hand, so a failure here is only logged."""
        try:
            await self._call(f"rm -rf {quoted_dir}", timeout=15.0)
        except PodmanContainerLostError:
            raise
        except PodmanGatewayError as e:
            logger.warning(f"gateway job-dir cleanup failed (ignored): {str(e)[:200]}")

    async def _keepalive_loop(self) -> None:
        """Refresh the binding's TTL while the container idles between turns."""
        interval = _settings().gateway_keepalive_s
        while True:
            await asyncio.sleep(interval)
            if self._lock.locked() or time.monotonic() - self._last_request < interval / 2:
                continue  # a command is running, or ran recently, which refreshes the TTL itself
            async with self._lock:
                try:
                    await self._call("true", timeout=15.0)
                except PodmanGatewayError as e:  # a lost container surfaces on the episode's next command
                    logger.warning(f"gateway keepalive failed: {e}")

    # ------------------------------------------------------------------ file IO

    async def _put_archive(self, tar_bytes: bytes) -> None:
        """Extract a tar at ``/`` in the container, in idempotent base64 chunks when it is large."""
        async with self._lock:
            try:
                if len(tar_bytes) <= _UPLOAD_CHUNK_BYTES:
                    result = await self._call(
                        "base64 -d | tar -xf - -C /", stdin=base64.b64encode(tar_bytes).decode(), timeout=45.0
                    )
                    _check(result, "archive upload")
                    return
                stage = shlex.quote(f"{_WORK_DIR}/upload_{uuid.uuid4().hex[:16]}")
                _check(await self._call(f"mkdir -p {stage}", timeout=15.0), "upload staging")
                for index in range(0, len(tar_bytes), _UPLOAD_CHUNK_BYTES):
                    part = index // _UPLOAD_CHUNK_BYTES
                    result = await self._call(
                        f"base64 -d > {stage}/part.{part:06d}",
                        stdin=base64.b64encode(tar_bytes[index : index + _UPLOAD_CHUNK_BYTES]).decode(),
                        timeout=30.0,
                    )
                    _check(result, f"upload chunk {part}")
                result = await self._call(
                    f"cat {stage}/part.* | tar -xf - -C / && rm -rf {stage}", timeout=_WAIT_EXEC_TIMEOUT_S
                )
                _check(result, "upload assembly")
            except PodmanContainerLostError as e:
                raise SandboxLostError(f"sandbox {self._image} lost: {_describe_loss(e)}", reason=e.reason) from e
            except PodmanGatewayError as e:
                if e.status_code == 404:
                    raise SandboxLostError(f"sandbox {self._image} unreachable: {e}", reason="binding_lost") from e
                raise SandboxInfraError(f"gateway upload failed: {e}") from e


def _check(result: CommandResult, what: str) -> None:
    if result.exit_code != 0:
        raise PodmanGatewayError(f"{what} failed (exit={result.exit_code}): {result.stderr[:500]}")


def _describe_loss(error: PodmanContainerLostError) -> str:
    measured = f" ({error.observed} {error.unit or ''} > {error.limit})" if error.observed is not None else ""
    return f"{error.reason}{measured}"


async def _retry(call: Callable[[], Awaitable], attempts: int):
    """Run ``call``, retrying transient gateway failures; every request this backend sends is safe to repeat."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except PodmanContainerLostError:
            raise
        except PodmanGatewayError as e:
            permanent = e.status_code is not None and e.status_code not in _TRANSIENT_STATUSES
            if permanent or attempt == attempts:
                raise
            delay = min(_RETRY_BASE_S * 2 ** (attempt - 1), _RETRY_MAX_S) + random.uniform(0.0, 0.5)
            logger.warning(f"gateway request failed ({e.status_code}); retry {attempt}/{attempts - 1} in {delay:.1f}s")
            await asyncio.sleep(delay)


async def close_live_sandboxes() -> int:
    """Close every container this rollout's episodes hold on the fleet (the abort hook). Returns the count."""
    ids = [path.name for path in _live_dir().iterdir()]
    async with _client() as client:
        for affinity_id in ids:
            (_live_dir() / affinity_id).unlink(missing_ok=True)
            try:
                await client.close(affinity_id)
            except PodmanGatewayError as e:  # one already lost counts as closed too
                logger.debug(f"closing gateway sandbox {affinity_id[:12]}: {e}")
    return len(ids)
