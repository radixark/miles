from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import IO

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import polling
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import (
    ContainerKey,
    ContainerRun,
    container_runs,
    selected_pods,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.polling import polling_in_background

logger = logging.getLogger(__name__)

_STOP_GRACE_SECONDS = 5.0
_ERROR_TAIL_CHARACTERS = 500


@contextmanager
def with_log_following(*, namespace: str, selector: str) -> Iterator[None]:
    follower = _LogFollower(namespace=namespace, selector=selector)
    try:
        with polling_in_background(
            follower.reconcile,
            description="look for pods to follow",
            join_timeout=polling.POLL_INTERVAL_SECONDS + _STOP_GRACE_SECONDS,
        ):
            yield
    finally:
        follower.stop_all_streams()


class _LogFollower:
    def __init__(self, *, namespace: str, selector: str) -> None:
        self._namespace = namespace
        self._selector = selector
        self._streams: dict[ContainerKey, _LogStream] = {}
        self._followed_container_ids: set[str] = set()

    def reconcile(self) -> None:
        runs = container_runs(selected_pods(self._namespace, self._selector))
        for key, stream in list(self._streams.items()):
            if (run := runs.get(key)) is None or run.container_id != stream.container_id:
                self._streams.pop(key).stop()

        for key, run in runs.items():
            if stream := self._streams.get(key):
                stream.resume_if_dropped(run)
            elif run.container_id not in self._followed_container_ids:
                self._followed_container_ids.add(run.container_id)
                self._streams[key] = _LogStream(namespace=self._namespace, run=run)

    def stop_all_streams(self) -> None:
        stopping = []
        while self._streams:
            _, stream = self._streams.popitem()
            stream.ask_to_stop()
            stopping.append(stream)

        for stream in stopping:
            stream.wait_until_stopped()


class _LogStream:
    def __init__(self, *, namespace: str, run: ContainerRun) -> None:
        self._namespace = namespace
        self._run = run
        self._prefix = f"[{run.key.pod}/{run.key.container}{' (previous)' if run.key.previous else ''}]"
        self._last_timestamp: str | None = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._start()

    @property
    def container_id(self) -> str:
        return self._run.container_id

    def resume_if_dropped(self, run: ContainerRun) -> None:
        self._run = run
        if self._stopped or not run.running:
            return
        if self._thread is not None and not self._thread.is_alive():
            self._start()

    def stop(self) -> None:
        self.ask_to_stop()
        self.wait_until_stopped()

    def ask_to_stop(self) -> None:
        self._stopped = True
        if (process := self._process) is not None and process.poll() is None:
            process.terminate()

    def wait_until_stopped(self) -> None:
        if (process := self._process) is not None:
            try:
                process.wait(timeout=_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_GRACE_SECONDS)

    def _start(self) -> None:
        command = Kubectl.logs_command(
            namespace=self._namespace,
            target=self._run.key.pod,
            container=self._run.key.container,
            follow=self._run.running,
            previous=self._run.key.previous,
            since_time=self._last_timestamp,
        )
        errors = tempfile.TemporaryFile(mode="w+")
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        except OSError as error:
            logger.warning(f"{self._prefix} could not be followed ({error})")
            self._stopped = True
            errors.close()
            return

        self._thread = threading.Thread(target=self._read_until_done, args=(self._process, errors), daemon=True)
        self._thread.start()

    def _read_until_done(self, process: subprocess.Popen, errors: IO[str]) -> None:
        for line in process.stdout:
            self._emit(line.rstrip("\n"))
        process.wait()

        if process.returncode and not self._stopped:
            self._stopped = True
            errors.seek(0)
            logger.warning(f"{self._prefix} stopped: {errors.read()[-_ERROR_TAIL_CHARACTERS:].strip()}")
        errors.close()

    def _emit(self, line: str) -> None:
        timestamp, _, rest = line.partition(" ")
        if _is_timestamp(timestamp):
            self._last_timestamp = timestamp
            line = rest
        logger.info(f"{self._prefix} {line}")


def _is_timestamp(token: str) -> bool:
    try:
        datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
