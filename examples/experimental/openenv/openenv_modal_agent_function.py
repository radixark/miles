"""Modal-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN Modal sandbox — built from the
task's OFFICIAL image plus an env server layer, terminated when the episode
ends.

The agent loop and training wrapper live in ``openenv_agent_function``, and
everything about running episodes on per-episode sandboxes is
``openenv_sandbox_common``'s ``SandboxBackend``. This module is only what is
Modal-specific: how one sandbox comes into being, which of its SDK's errors are
worth retrying, and its knobs. The image recipe lives in ``tb2_sandbox_recipe``
and its Modal materialization in ``tb2_sandbox_modal``; like the other sandbox
backends, this variant needs the pinned tbench2_env install from the README
(canonical test.sh scoring and verifier-asset withholding built into the
server).

Credentials are the one place Modal does not fit the other backends' shape: there
is no single API key, so nothing here reads or forwards one. The SDK resolves
MODAL_TOKEN_ID + MODAL_TOKEN_SECRET from the worker's own environment, else the
config file at MODAL_CONFIG_PATH (default ``~/.modal.toml``) — and the launcher
forwards that PATH, never the token, on the same reasoning as the other
providers' key files (see openenv_launch_common).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. The first
                    episode of a task pays its image build; repeats hit
                    Modal's layer cache, and all tasks share every layer but
                    the last (see tb2_sandbox_modal.task_image).
  MODAL_TOKEN_ID / MODAL_TOKEN_SECRET / MODAL_CONFIG_PATH   credential supply,
                    read by the SDK itself. MODAL_PROFILE / MODAL_ENVIRONMENT
                    select a profile / workspace environment when set.
  OPENENV_MODAL_APP                app the sandboxes are created under
                    (default openenv-tbench2) — what a sweep scopes to.
  OPENENV_MODAL_CREATE_CONCURRENCY max in-flight sandbox creates (default 4).
  OPENENV_MODAL_CREATE_MAX_RETRIES how many throttled creates to retry (default 8).

Everything describing ONE sandbox — its app, lifetime, and the create/ready
deadlines — is documented and read in ``tb2_sandbox_modal``, and the per-exec
timeout inside it (TB2_COMMAND_TIMEOUT_S) in ``tb2_sandbox_recipe``, next to
the code each one steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_modal

logger = logging.getLogger(__name__)


# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1012 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Modal caps concurrent containers per plan; the shared backend caps in-flight
# creates process-wide and retries throttled ones with jittered exponential
# backoff (knobs: OPENENV_MODAL_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Modal was out of room.

    Modal types a hit ceiling (the plan's container concurrency, or a workspace
    resource limit) as ResourceExhaustedError; the text match covers the same
    condition surfacing as a message from an older SDK or a proxy.
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where modal is already in sys.modules,
    # having just raised `exc`), and offline tests must not require the SDK.
    try:
        from modal.exception import ResourceExhaustedError

        if isinstance(exc, ResourceExhaustedError):
            return True
    except ImportError:  # pragma: no cover - only without the modal SDK
        pass
    return common.throttle_text(exc, "resource exhausted")


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    sandbox, url = tb2_sandbox_modal.create_task_sandbox(Path(tasks_dir) / task_id)
    return sandbox.terminate, url


BACKEND = common.SandboxBackend(
    provider="Modal",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("MODAL"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
