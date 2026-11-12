"""A sandbox start function on E2B, or any service speaking its API (the ``e2b`` SDK reads ``E2B_API_URL``).

    --custom-agent-function-mode worker
    --custom-agent-function-worker examples.experimental.agent_sandbox.e2b_start.start

The template's start command waits for ``BOOT_CONFIG`` (``{"env": {...}}``), adds that env, and serves
``python -m miles.rollout.agentic.worker`` on ``WORKER_PORT``. The boot env holds the E2B key, for the sandboxes the
agent creates, the strings in ``SANDBOX_WORKER_BOOT_ENV_FILE``, and this host's agent worker secret when it has one.
The worker is reached at ``get_host(WORKER_PORT)``.

Closing the sandbox waits for the worker's agent processes to exit, so a cancelled agent's own cleanup finishes, then
copies the ``SANDBOX_WORKER_RECORDS`` files of each dir under ``SANDBOX_WORKER_RECORDS_DIR`` to the same path on this
host, then kills it.

Env vars:
  SANDBOX_WORKER_TEMPLATE        the template (required)
  SANDBOX_WORKER_TIMEOUT_S       the sandbox's lifetime, so one never killed still expires (default 7800)
  SANDBOX_WORKER_OWNER           its owner metadata (default: this host's name)
  SANDBOX_WORKER_BOOT_ENV_FILE   a JSON object of strings added to the boot env, by path so secrets stay out of logs
  SANDBOX_WORKER_RECORDS_DIR     where the agent leaves its records, e.g. HARBOR_TRIALS_DIR's value (default: none)
  SANDBOX_WORKER_RECORDS         comma-separated paths under each of its dirs
                                 (default result.json,config.json,trial.log,agent/timings.jsonl)
  SANDBOX_WORKER_URL_SCHEME      the worker URL's scheme (default https)
"""

import functools
import json
import logging
import os
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miles.rollout.agentic.agent_function import _CANCEL_GRACE_S, _LEFTOVER_CLEANUP_S
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, resolve_provider_api_key
from miles.rollout.agentic.remote import SECRET_ENV_VAR, worker_secret

logger = logging.getLogger(__name__)

BOOT_CONFIG = "/tmp/agent_worker_boot.json"
WORKER_PORT = 11100
AGENT_PROCESS_PROBE = "grep -qsa '[_]child_main' /proc/[0-9]*/cmdline"
_DEFAULT_RECORDS = "result.json,config.json,trial.log,agent/timings.jsonl"
_KILL_ATTEMPTS = 3


def start(task_id: str, tasks_dir: str) -> tuple[Callable[[], None], str]:
    from e2b import Sandbox

    env = boot_env()
    sandbox = Sandbox.create(
        template=os.environ["SANDBOX_WORKER_TEMPLATE"],
        timeout=int(os.getenv("SANDBOX_WORKER_TIMEOUT_S", "7800")),
        metadata={
            "owner": os.getenv("SANDBOX_WORKER_OWNER") or socket.gethostname(),
            "role": "agent-worker",
            "call": task_id[:63],
        },
        api_key=env["E2B_API_KEY"],
    )
    try:
        sandbox.files.write(BOOT_CONFIG, json.dumps({"env": env}))
    except Exception:
        _kill(sandbox, task_id)
        raise
    scheme = os.getenv("SANDBOX_WORKER_URL_SCHEME", "https")
    return functools.partial(_close, sandbox, task_id), f"{scheme}://{sandbox.get_host(WORKER_PORT)}"


def boot_env() -> dict[str, str]:
    spec = PROVIDER_CREDENTIALS["e2b"]
    (key_env,) = spec["key_env_vars"]
    env = {key_env: resolve_provider_api_key(key_env, spec["file_env_var"], spec["default_path"])}
    if path := os.getenv("SANDBOX_WORKER_BOOT_ENV_FILE"):
        extra = json.loads(Path(path).expanduser().read_text())
        if not isinstance(extra, dict) or not all(isinstance(v, str) for v in extra.values()):
            raise ValueError(f"{path} must hold a JSON object of strings")
        env.update(extra)
    if secret := worker_secret():
        env[SECRET_ENV_VAR] = secret
    return env


def _close(sandbox: Any, task_id: str) -> None:
    _wait_for_agent_exit(sandbox, task_id)
    _copy_records(sandbox)
    _kill(sandbox, task_id)


def _wait_for_agent_exit(sandbox: Any, task_id: str) -> None:
    deadline_s = int(_CANCEL_GRACE_S + _LEFTOVER_CLEANUP_S) + 10
    try:
        sandbox.commands.run(
            f"timeout {deadline_s} bash -c 'while {AGENT_PROCESS_PROBE}; do sleep 0.5; done'", timeout=deadline_s + 30
        )
    except Exception as exc:
        logger.warning(f"agent processes in sandbox {sandbox.sandbox_id} for {task_id} not seen to exit: {exc!r}")


def _copy_records(sandbox: Any) -> None:
    from e2b import FileType

    root = os.getenv("SANDBOX_WORKER_RECORDS_DIR", "").rstrip("/")
    if not root:
        return
    names = [name for name in os.getenv("SANDBOX_WORKER_RECORDS", _DEFAULT_RECORDS).split(",") if name]
    try:
        dirs = [entry.name for entry in sandbox.files.list(root) if entry.type == FileType.DIR]
    except Exception as exc:
        logger.warning(f"records under {root} not listed in sandbox {sandbox.sandbox_id}: {exc!r}")
        return
    for path in (f"{root}/{d}/{name}" for d in dirs for name in names):
        try:
            data = sandbox.files.read(path, format="bytes")
        except Exception as exc:
            logger.warning(f"record {path} not copied from sandbox {sandbox.sandbox_id}: {exc!r}")
            continue
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(bytes(data))


def _kill(sandbox: Any, task_id: str) -> None:
    for attempt in range(_KILL_ATTEMPTS):
        try:
            sandbox.kill()
            return
        except Exception as exc:
            logger.warning(f"killing sandbox {sandbox.sandbox_id} for {task_id} (attempt {attempt + 1}): {exc!r}")
            time.sleep(2 * (attempt + 1))
    logger.error(f"sandbox {sandbox.sandbox_id} for {task_id} left to expire at its timeout")
