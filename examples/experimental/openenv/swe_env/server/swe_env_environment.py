# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-bench environment server implementation (Docker-isolated).

Each episode runs inside the task's own SWE-bench container. Unlike the
TB2 docker env, SWE-bench tasks ship no pre-built ``docker_image`` in
task.toml -- they carry an ``environment/Dockerfile`` (``FROM
swebench/sweb.eval.x86_64.<id>``). This env builds that Dockerfile once per
task (tagged ``openenv-swe:<task_id>`` and cached) and runs the container at
the repo checkout ``/testbed``.

Evaluation defers entirely to the harbor-format ``tests/test.sh`` contract:
the task's ``tests/`` dir is copied into the container at ``/tests`` and
``bash /tests/test.sh`` runs it (conda activate testbed -> reinstall repo ->
apply swebench test patch -> pytest -> swebench parser). test.sh exits 0 iff
the task is resolved, which becomes reward 1.0.

Requires:
- Docker socket reachable (the server runs on a Docker host).
- Disk for the SWE-bench base images (~1-2 GB each).
"""

from __future__ import annotations

import io
import logging
import os
import re
import tarfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment


# Support both in-repo and standalone imports
try:
    from swe_env.models import SweAction, SweObservation, SweState
except ImportError:
    from models import SweAction, SweObservation, SweState


logger = logging.getLogger(__name__)

# Container path the SWE-bench harbor test.sh expects its harness files at
# (parser.py reads /tests/config.json).
_TESTS_MOUNT = "/tests"
# Repo checkout inside every swebench eval image.
_WORKDIR = "/testbed"
_EXIT_MARKER = "__SWE_EXIT_CODE__"


def _read_instruction(task_dir: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text(encoding="utf-8")
    return ""


def _image_tag(task_id: str) -> str:
    """A valid docker tag for a task id (lowercase repo + sanitized tag)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", task_id)
    return f"openenv-swe:{safe}"


class SweDockerEnvironment(Environment[SweAction, SweObservation, SweState]):
    """OpenEnv wrapper around SWE-bench Verified tasks with Docker isolation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float = 600.0,
        default_task_id: str | None = None,
    ) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("SWE_TASKS_DIR", "")
        self.output_dir = Path(
            output_dir or os.getenv("SWE_OUTPUT_DIR", "/tmp/swe_env_runs")
        )
        self.command_timeout_s = command_timeout_s
        self.default_task_id = default_task_id or os.getenv("SWE_DEFAULT_TASK_ID", "")

        self._state = SweState()
        self._task_dir: Path | None = None
        self._docker_client = None
        self._container = None
        self._instruction = ""
        self._task_image = ""

    def _get_docker_client(self) -> Any:
        if self._docker_client is None:
            try:
                import docker

                self._docker_client = docker.from_env()
            except Exception as exc:
                raise RuntimeError(
                    f"Docker client not available. Ensure Docker is reachable. Error: {exc}"
                ) from exc
        return self._docker_client

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SweObservation:
        del seed

        task_id = (
            kwargs.get("task_id") or kwargs.get("task_name") or self.default_task_id
        )
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        self._instruction = _read_instruction(task_dir)
        self._task_dir = task_dir

        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        trial_dir = self.output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        self._task_image = self._build_image(task_dir, resolved_task_id)
        self._start_container(task_dir)

        self._state = SweState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=resolved_task_id,
            task_path=str(task_dir),
            terminal_ready=True,
        )

        return SweObservation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={"docker_image": self._task_image},
            reward=0.0,
            done=False,
        )

    def _build_image(self, task_dir: Path, task_id: str) -> str:
        """Build (or reuse a cached) image from the task's environment/Dockerfile."""
        docker = self._get_docker_client()
        tag = _image_tag(task_id)
        try:
            docker.images.get(tag)
            return tag
        except Exception:
            pass

        env_dir = task_dir / "environment"
        if not (env_dir / "Dockerfile").exists():
            raise FileNotFoundError(f"No environment/Dockerfile under {task_dir}")

        logger.info("Building SWE image %s from %s ...", tag, env_dir)
        docker.images.build(path=str(env_dir), tag=tag, rm=True, forcerm=True)
        return tag

    def _start_container(self, task_dir: Path) -> None:
        docker = self._get_docker_client()
        try:
            self._container = docker.containers.run(
                image=self._task_image,
                command="sleep infinity",
                detach=True,
                network_mode="host",
                working_dir=_WORKDIR,
                remove=False,
            )
            # Stage the harbor test harness where test.sh/parser.py expect it.
            self._container.exec_run(cmd=f"mkdir -p {_TESTS_MOUNT}")
            self._copy_dir_to_container(task_dir / "tests", _TESTS_MOUNT)
        except Exception as exc:
            raise RuntimeError(f"Failed to start SWE container: {exc}") from exc

    def _copy_dir_to_container(self, src_dir: Path, dest_path: str) -> None:
        if self._container is None:
            raise RuntimeError("Container not started")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for item in src_dir.rglob("*"):
                tar.add(str(item), arcname=str(item.relative_to(src_dir)))
        tar_stream.seek(0)
        self._container.put_archive(dest_path, tar_stream.getvalue())

    def _exec_in_container(
        self, command: str, workdir: str = _WORKDIR
    ) -> tuple[int, str]:
        if self._container is None:
            raise RuntimeError("Container not started. Call reset() first.")
        exit_code, output = self._container.exec_run(
            cmd=["bash", "-c", f"cd {workdir} && {command}"],
            stdout=True,
            stderr=True,
        )
        return exit_code, output.decode("utf-8", errors="replace")

    def step(
        self,
        action: SweAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SweObservation:
        del timeout_s, kwargs

        if not isinstance(action, SweAction):
            raise TypeError(f"Expected SweAction, got {type(action)}")
        if self._task_dir is None:
            raise RuntimeError("SWE environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "swe-session"

        try:
            if action.action_type == "exec":
                exit_code, output = self._exec_in_container(action.command)
                success = exit_code == 0
            elif action.action_type == "write_file":
                self._write_file(action.file_path, action.content)
                output = f"Wrote to {action.file_path}"
            elif action.action_type == "evaluate":
                output, reward, info = self._evaluate()
                done = True
            elif action.action_type == "close":
                self.close()
                output = "Closed SWE environment."
                done = True
            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")
        except Exception as exc:
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id

        return SweObservation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id,
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    def _write_file(self, file_path: str, content: str) -> None:
        if self._container is None:
            raise RuntimeError("Container not started.")
        # Stream the bytes via tar so arbitrary content (quotes, heredoc markers)
        # survives intact rather than going through a fragile shell heredoc.
        dest = Path(file_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=dest.name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        parent = str(dest.parent) if str(dest.parent) else "/"
        self._container.exec_run(cmd=["mkdir", "-p", parent])
        self._container.put_archive(parent, tar_stream.getvalue())

    def _evaluate(self) -> tuple[str, float, dict[str, Any]]:
        if self._container is None:
            raise RuntimeError("Container not started.")

        # parser.py writes /logs/verifier/report.json before test.sh's own
        # mkdir runs, so create the dir up front (harbor's runner does this).
        cmd = f"mkdir -p /logs/verifier && bash {_TESTS_MOUNT}/test.sh; echo {_EXIT_MARKER}:$?"
        exit_code, output = self._container.exec_run(
            cmd=["bash", "-c", cmd],
            stdout=True,
            stderr=True,
        )
        output_str = output.decode("utf-8", errors="replace")

        ec = self._parse_exit_code(output_str, fallback=int(exit_code or 1))
        reward = 1.0 if ec == 0 else 0.0
        info = {"tests_passed": ec == 0, "exit_code": ec}
        return output_str, reward, info

    @staticmethod
    def _parse_exit_code(output: str, fallback: int) -> int:
        for line in output.splitlines()[::-1]:
            if _EXIT_MARKER in line:
                try:
                    return int(line.split(":", 1)[1].strip())
                except Exception:
                    return fallback
        return fallback

    @property
    def state(self) -> SweState:
        return self._state

    def close(self) -> None:
        if self._container:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
        self._task_dir = None
        self._instruction = ""

    def _resolve_task_path(self, task_id: str | None, task_path: str | None) -> Path:
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset SWE environment.")
        if not self.tasks_dir:
            raise ValueError("SWE_TASKS_DIR is not set; cannot resolve task_id.")

        resolved = Path(self.tasks_dir).expanduser().resolve() / task_id
        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved
