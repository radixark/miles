"""The gateway sandbox backend: TMax containers on a LiteRegistry podman fleet.

A port of open-instruct's ``GatewayBackend`` (branch ``literegistry_podman``,
``open_instruct/environments/backends.py``) onto asyncio. The fleet is a separate deployment
(https://github.com/goncalorafaria/literegistry, ``literegistry_podman_beaker``): rootless podman replicas and
Docker Hub mirrors that heartbeat into a Redis registry, behind a gateway that routes each container's requests to
the replica that owns it. The training job runs no podman, so no sandbox shares a node with Megatron or SGLang.

Per episode:

* ``POST /affinity/handshake {service, image}``: the gateway picks a replica, which starts a container from the task
  image (pulled through the fleet's mirrors) and binds its ``affinity_id`` to itself.
* ``POST /affinity/podman {affinity_id, command, stdin, timeout, workdir}``: one ``podman exec`` on that replica.
  Replicas cap an exec at 60 s and 1 MB of stdin, and abort (the gateway then retries, re-running the command)
  rather than truncate more than 1 MB of stdout or 256 KB of stderr. So a command never runs as a bare exec: one exec
  launches it detached with its output in files, then wait execs block in the container until it finishes and
  return its exit code and bounded output. A ``mkdir`` lock makes a retried launch a no-op, and every upload chunk is
  its own file, so gateway retries are safe.
* ``POST /affinity/close {affinity_id}``: removes the container.

The binding expires after 15 idle minutes, so a keepalive exec runs while the container idles through a long
generation. A binding that is lost anyway (the replica died, or the gateway expired it) is re-handshaken once and the
command retried, losing the container's state, as the podman backend recreates a vanished container once. A
container the replica removed or stopped under the episode (its resource watchdog, or the agent killing PID 1) ends
the episode with reward 0 (``SandboxLostError``). Replicas set no memory limit.
"""

import asyncio
import base64
import logging
import random
import shlex
import tempfile
import time
import uuid
from pathlib import Path

import httpx
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

_SERVICE = "podman"
# The replica's per-exec contract (literegistry PodmanRequest / PodmanAffinityConfig).
_SERVER_EXEC_TIMEOUT_CAP_S = 60.0
# Base64 inflates by 4/3; stay well below the 1 MB stdin cap.
_UPLOAD_CHUNK_BYTES = 512 * 1024
# Well below the replica's 1 MB stdout / 256 KB stderr caps, which abort the exec rather than truncate.
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
_HANDSHAKE_TIMEOUT_S = 240.0
_REQUEST_ATTEMPTS = 4
_RETRY_BASE_S = 0.5
_RETRY_MAX_S = 8.0
# 503 texts for a binding whose replica is gone: "no longer registered" once it left the roster, "server is
# unavailable" while it is still listed but refuses connections. After this many in a row the session is lost.
_OWNER_LOST_MARKERS = ("no longer registered", "server is unavailable")
_OWNER_LOST_ATTEMPTS = 3
_CONTAINER_REMOVED_MARKER = "container_not_found"
_CONTAINER_STOPPED_MARKERS = ("container state improper", "can only create exec sessions on running containers")


class GatewayError(RuntimeError):
    """A gateway request failed or returned an invalid response."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class GatewaySessionLostError(GatewayError):
    """The container's affinity binding no longer exists on the gateway."""


def _live_dir() -> Path:
    """This rollout's live containers, one empty file per affinity id, for the abort hook."""
    path = Path(tempfile.gettempdir()) / f"miles-tmax-gateway-live-{current_scope()}"
    path.mkdir(exist_ok=True)
    return path


class TmaxGatewaySandbox(SandboxBase):
    """The gateway backend: a container on a LiteRegistry podman replica."""

    def __init__(self, image: str, task_dir: str):
        super().__init__(image, task_dir)
        if not _settings().gateway_url:  # a configuration error, not an infra failure to score as 0
            raise ValueError("the gateway sandbox backend needs TMAX_GATEWAY_URL")
        self._http: httpx.AsyncClient | None = None
        self._affinity_id: str | None = None
        self._instance_id: str | None = None
        self._ready = False  # the episode's container is up; a lost binding is re-handshaken only after that
        self._restarted = False
        self._lock = asyncio.Lock()  # one exec at a time per container; the keepalive skips while one runs
        self._last_request = time.monotonic()
        self._keepalive: asyncio.Task | None = None

    async def __aenter__(self) -> "TmaxGatewaySandbox":
        self._http = httpx.AsyncClient(timeout=None)
        last_error: Exception | None = None
        for attempt in range(_START_ATTEMPTS):
            try:
                await self._start()
                await self._prepare_runtime()
                self._ready = True
                if _settings().gateway_keepalive_s > 0:
                    self._keepalive = asyncio.create_task(self._keepalive_loop())
                return self
            except asyncio.CancelledError:
                await asyncio.shield(self._release())
                raise
            except Exception as e:
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
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    # ------------------------------------------------------------------ container lifecycle

    async def _start(self) -> None:
        async with rollout_semaphore("tmax-gateway-start", _settings().start_concurrency):
            result = await _post(
                self._http,
                "affinity/handshake",
                {"service": _SERVICE, "image": self._image, "client_id": f"miles-tmax-{uuid.uuid4().hex[:12]}"},
                request_timeout=_HANDSHAKE_TIMEOUT_S,
                attempts=_HANDSHAKE_ATTEMPTS,
            )
        affinity_id = result.get("affinity_id")
        if not isinstance(affinity_id, str) or not affinity_id:
            raise GatewayError(f"gateway handshake returned no affinity_id: {result}")
        self._affinity_id, self._instance_id = affinity_id, result.get("instance_id")
        (_live_dir() / affinity_id).touch()

    async def _close_container(self) -> None:
        affinity_id, self._affinity_id = self._affinity_id, None
        if affinity_id is None:
            return
        (_live_dir() / affinity_id).unlink(missing_ok=True)
        try:
            await _post(
                self._http,
                "affinity/close",
                {"service": _SERVICE, "affinity_id": affinity_id},
                request_timeout=60.0,
                attempts=2,
            )
        except GatewaySessionLostError:
            pass
        except GatewayError as e:  # the replica reaps leftover containers on restart; never fail the episode on it
            logger.warning(f"failed to close gateway sandbox {affinity_id[:12]}: {e}")

    async def _restart(self, error: BaseException) -> None:
        logger.warning(
            f"gateway sandbox for {self._image} lost its binding ({error}); re-handshaking once, state is lost"
        )
        self._restarted = True
        await self._close_container()
        try:
            await self._start()
            await self._prepare_runtime()
        except Exception as e:
            raise SandboxInfraError(f"sandbox restart failed: {e!r}") from e

    # ------------------------------------------------------------------ commands

    async def _exec(self, command: str, timeout_s: int | None = None) -> ExecResult:
        timeout_s = timeout_s or _settings().command_timeout_s
        try:
            return await self._run_command(command, timeout_s)
        except GatewaySessionLostError as e:
            if _CONTAINER_REMOVED_MARKER in e.body:
                raise SandboxLostError(f"sandbox {self._image} was removed by its replica: {e.body[:300]}") from e
            if not self._ready or self._restarted:
                raise SandboxInfraError(f"gateway sandbox lost: {e}") from e
            await self._restart(e)
            return await self._run_command(command, timeout_s)
        except GatewayError as e:
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
        # The job dir is the lock: a gateway retry of a launch that already ran finds it and only reports success.
        # `{ ... & }` backgrounds just the setsid command, so `cat` still reads the request's stdin.
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
            launch = await self._raw_exec(launcher, stdin=command, exec_timeout=30.0)
            if "LAUNCHED" not in launch.get("stdout", ""):
                self._raise_if_stopped(launch)
                raise GatewayError(
                    f"command launcher failed (exit={launch.get('exit_code')}): {launch.get('stderr', '')[:500]}"
                )
            # The command's own timeout, plus TERM -> KILL grace and slack for the launcher.
            deadline = time.monotonic() + timeout_s + 60.0
            markerless = 0
            while True:
                response = await self._raw_exec(waiter, exec_timeout=_WAIT_EXEC_TIMEOUT_S)
                stdout = response.get("stdout", "")
                if _DONE_MARKER in stdout:
                    break
                if _PENDING_MARKER not in stdout:
                    markerless += 1
                    self._raise_if_stopped(response)
                    if markerless >= _MAX_MARKERLESS_WAITS:
                        await self._probe_alive()
                        raise GatewayError(
                            f"wait exec returned no marker {markerless}x (exit={response.get('exit_code')}): "
                            f"stdout={stdout[:200]!r} stderr={response.get('stderr', '')[:200]!r}"
                        )
                if time.monotonic() > deadline:
                    await self._cleanup(quoted_dir)
                    raise GatewayError(f"command did not finish within {timeout_s}s plus grace")
            await self._cleanup(quoted_dir)
        rc_line, _, out = stdout.split(_DONE_MARKER, 1)[1].lstrip("\n").partition("\n")
        try:
            exit_code = int(rc_line.removeprefix("RC=").strip())
        except ValueError:
            raise GatewayError(f"wait exec returned an invalid rc line: {rc_line!r}") from None
        stderr = response.get("stderr", "")
        if exit_code == 124:
            stderr = f"Command timed out after {timeout_s}s.\n{stderr}"
        return ExecResult(stdout=out, stderr=stderr, exit_code=exit_code)

    async def _raw_exec(self, command: str, stdin: str = "", exec_timeout: float = 30.0) -> dict:
        """One bounded ``podman exec`` on the container's replica."""
        if self._affinity_id is None:
            raise GatewaySessionLostError("no container")
        exec_timeout = min(exec_timeout, _SERVER_EXEC_TIMEOUT_CAP_S - 1.0)
        self._last_request = time.monotonic()
        return await _post(
            self._http,
            "affinity/podman",
            {
                "service": _SERVICE,
                "affinity_id": self._affinity_id,
                "command": command,
                "stdin": stdin,
                "timeout": exec_timeout,
                "workdir": "/",
            },
            # The replica allows exec_timeout + 5 s; leave room for queueing in the gateway.
            request_timeout=exec_timeout + 20.0,
        )

    def _raise_if_stopped(self, response: dict) -> None:
        stderr = response.get("stderr", "") or ""
        if any(marker in stderr for marker in _CONTAINER_STOPPED_MARKERS):
            raise SandboxLostError(f"sandbox {self._image} stopped (its init process died): {stderr[:300]}")

    async def _probe_alive(self) -> None:
        try:
            probe = await self._raw_exec("true", exec_timeout=15.0)
        except GatewaySessionLostError as e:
            raise SandboxLostError(f"sandbox {self._image} is gone: {e.body[:300]}") from e
        self._raise_if_stopped(probe)

    async def _cleanup(self, quoted_dir: str) -> None:
        """Remove a finished command's scratch dir; the result is in hand, so a failure here is only logged."""
        try:
            await self._raw_exec(f"rm -rf {quoted_dir}", exec_timeout=15.0)
        except GatewayError as e:
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
                    await self._raw_exec("true", exec_timeout=15.0)
                except GatewayError as e:
                    logger.warning(f"gateway keepalive failed for {(self._affinity_id or '?')[:12]}: {e}")

    # ------------------------------------------------------------------ file IO

    async def _put_archive(self, tar_bytes: bytes) -> None:
        """Extract a tar at ``/`` in the container, in idempotent base64 chunks when it is large."""
        async with self._lock:
            try:
                if len(tar_bytes) <= _UPLOAD_CHUNK_BYTES:
                    result = await self._raw_exec(
                        "base64 -d | tar -xf - -C /", stdin=base64.b64encode(tar_bytes).decode(), exec_timeout=45.0
                    )
                    _check(result, "archive upload")
                    return
                stage = shlex.quote(f"{_WORK_DIR}/upload_{uuid.uuid4().hex[:16]}")
                _check(await self._raw_exec(f"mkdir -p {stage}", exec_timeout=15.0), "upload staging")
                for index in range(0, len(tar_bytes), _UPLOAD_CHUNK_BYTES):
                    chunk = tar_bytes[index : index + _UPLOAD_CHUNK_BYTES]
                    result = await self._raw_exec(
                        f"base64 -d > {stage}/part.{index // _UPLOAD_CHUNK_BYTES:06d}",
                        stdin=base64.b64encode(chunk).decode(),
                        exec_timeout=30.0,
                    )
                    _check(result, f"upload chunk {index // _UPLOAD_CHUNK_BYTES}")
                result = await self._raw_exec(
                    f"cat {stage}/part.* | tar -xf - -C / && rm -rf {stage}", exec_timeout=_WAIT_EXEC_TIMEOUT_S
                )
                _check(result, "upload assembly")
            except GatewaySessionLostError as e:
                if _CONTAINER_REMOVED_MARKER in e.body:
                    raise SandboxLostError(f"sandbox {self._image} was removed by its replica: {e.body[:300]}") from e
                raise SandboxInfraError(f"gateway sandbox lost during upload: {e}") from e
            except GatewayError as e:
                raise SandboxInfraError(f"gateway upload failed: {e}") from e


def _check(result: dict, what: str) -> None:
    if result.get("exit_code") != 0:
        raise GatewayError(f"{what} failed (exit={result.get('exit_code')}): {result.get('stderr', '')[:500]}")


async def _post(
    http: httpx.AsyncClient | None,
    endpoint: str,
    payload: dict,
    request_timeout: float,
    attempts: int = _REQUEST_ATTEMPTS,
) -> dict:
    """POST to the gateway with bounded retries; every request this backend sends is safe to repeat.

    Connection errors, timeouts and 408/500/502/503/504 are retried. 404 (binding expired) and 410 (the gateway
    found the binding's owner gone) raise ``GatewaySessionLostError``, as do repeated 503s saying the owner is
    unavailable or no longer registered: the gateway only drops a dead replica from its roster minutes later.
    """
    if http is None:
        raise GatewayError("gateway client closed")
    url = f"{_settings().gateway_url}/{endpoint}"
    last_error: GatewayError | None = None
    owner_lost = 0
    for attempt in range(1, attempts + 1):
        try:
            response = await http.post(url, json=payload, timeout=request_timeout)
        except httpx.HTTPError as e:
            last_error = GatewayError(f"gateway request to {endpoint} failed: {e!r}")
        else:
            text = response.text
            if response.status_code in (404, 410):
                raise GatewaySessionLostError(
                    f"gateway session lost ({endpoint}): {text[:500]}", status_code=response.status_code, body=text
                )
            if response.status_code == 503 and any(marker in text for marker in _OWNER_LOST_MARKERS):
                owner_lost += 1
                if owner_lost >= _OWNER_LOST_ATTEMPTS:
                    raise GatewaySessionLostError(
                        f"gateway owner gone ({endpoint}): {text[:500]}", status_code=503, body=text
                    )
            if response.status_code in (408, 500, 502, 503, 504):
                last_error = GatewayError(
                    f"gateway HTTP {response.status_code} for {endpoint}: {text[:500]}",
                    status_code=response.status_code,
                )
            elif response.status_code >= 400:
                raise GatewayError(
                    f"gateway HTTP {response.status_code} for {endpoint}: {text[:500]}",
                    status_code=response.status_code,
                )
            else:
                try:
                    result = response.json()
                except ValueError as e:
                    raise GatewayError(f"gateway returned non-JSON for {endpoint}: {e}") from e
                if not isinstance(result, dict):
                    raise GatewayError(f"gateway returned a non-object for {endpoint}")
                return result
        if attempt < attempts:
            await asyncio.sleep(min(_RETRY_BASE_S * 2 ** (attempt - 1), _RETRY_MAX_S) + random.uniform(0.0, 0.5))
    assert last_error is not None
    raise last_error


async def close_live_sandboxes() -> int:
    """Close every container this rollout's episodes hold on the fleet (the abort hook). Returns the count."""
    ids = [path.name for path in _live_dir().iterdir()]
    async with httpx.AsyncClient(timeout=None) as http:
        for affinity_id in ids:
            (_live_dir() / affinity_id).unlink(missing_ok=True)
            try:
                await _post(
                    http, "affinity/close", {"service": _SERVICE, "affinity_id": affinity_id}, 60.0, attempts=2
                )
            except GatewayError as e:
                logger.warning(f"cannot close gateway sandbox {affinity_id[:12]}: {e}")
    return len(ids)
