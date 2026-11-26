"""Runta-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN Runta runtime — restored from a
per-task checkpoint holding the task's OFFICIAL image plus an env server
layer, deleted when the episode ends.

The agent loop and training wrapper live in ``openenv_agent_function``
(sibling module) and are reused unchanged; this module only supplies its own
``run_episode`` — how an env comes into being (see the episode-wiring note
there). The image recipe lives in ``tb2_sandbox_recipe`` and its Runta
materialization in ``tb2_sandbox_runta``; like every sandbox backend, this one
needs the pinned tbench2_env install from the README (canonical test.sh
scoring and verifier-asset withholding built into the server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. Checkpoints
                    are built once per task under a recipe-digest name
                    (first episode of a task pays the build; repeats
                    restore), or pre-baked via the tb2_sandbox_runta CLI.
  RUNTA_TOKEN / RUNTA_TOKEN_FILE   token supply on the contract every backend
                    shares (file default ~/.config/runta/token; launchers
                    forward the PATH, never the value).
  RUNTA_ENDPOINT / RUNTA_INGRESS_BASE_URL   endpoint overrides: the first is
                    read by the SDK itself, the second says where published
                    ports are served when the endpoint does not imply it.
  OPENENV_RUNTA_CREATE_CONCURRENCY   max in-flight sandbox creates (default 4).
  OPENENV_RUNTA_CREATE_MAX_RETRIES   how many throttled creates to retry (default 8).

Everything describing ONE sandbox — its lifetime and the ready deadline — is
documented and read in ``tb2_sandbox_runta``, and the per-exec timeout inside it
(TB2_COMMAND_TIMEOUT_S) in ``tb2_sandbox_recipe``, next to the code each one
steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_runta

logger = logging.getLogger(__name__)


# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1025 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Providers rate-limit or run out of capacity under a fanned-out rollout; the
# shared backend caps in-flight creates per rollout and retries throttled ones
# with jittered exponential backoff (knobs: OPENENV_RUNTA_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Runta throttled it.

    The SDK raises one ApiError for every API failure and carries the HTTP
    status on it, so 429 is read off the status and nothing else is. The text
    match the other backends share is deliberately not consulted: a failed
    create quotes the runtime's UUID and the tail of a docker build log, and a
    "429" in either would have a deterministic build failure retried as
    throttling, holding the build lock and a create slot all the while.
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where runta is already in sys.modules,
    # having just raised `exc`), and offline tests must not require the SDK.
    try:
        from runta import ApiError
    except ImportError:  # without the SDK nothing raised is a Runta API error
        return False
    return isinstance(exc, ApiError) and exc.status_code == 429


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    runtime, url = tb2_sandbox_runta.create_task_sandbox(Path(tasks_dir) / task_id)
    # Not runtime.delete: the materialization's own close stops the runtime's
    # keepalive first, which a bare delete would race (see _Keepalive there).
    return (lambda: tb2_sandbox_runta.delete_task_sandbox(runtime)), url


BACKEND = common.SandboxBackend(
    provider="Runta",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("RUNTA"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
