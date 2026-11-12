"""boxd-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN `boxd <https://boxd.sh/?ref=miles>`_
machine — started from a snapshot of the task's OFFICIAL image plus an env
server layer, deleted when the episode ends.

The agent loop and training wrapper live in ``openenv_agent_function``, and
everything about running episodes on per-episode sandboxes is
``openenv_sandbox_common``'s ``SandboxBackend``. This module is only what is
boxd-specific: how one sandbox comes into being, which of its SDK's errors are
worth retrying, and its knobs. The image recipe lives in ``tb2_sandbox_recipe``
and its boxd materialization in ``tb2_sandbox_boxd``; like the other sandbox
backends, this variant needs the pinned tbench2_env install from the README
(canonical test.sh scoring and verifier-asset withholding built into the
server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. Snapshots
                    are baked once per task under a recipe-digest name
                    (first episode of a task pays the build; repeats start
                    from the snapshot), or pre-baked via the tb2_sandbox_boxd
                    CLI.
  BOXD_API_KEY / BOXD_API_KEY_FILE   key supply on the contract every backend
                    shares (file default ~/.config/boxd/api_key; launchers
                    forward the PATH, never the value).
  BOXD_REGION / BOXD_ORG      region and org, set up as in the Sandbox
                    Providers guide (docs/user-guide/sandbox-providers.md).
  OPENENV_BOXD_CREATE_CONCURRENCY   max in-flight machine creates (default 4).
  OPENENV_BOXD_CREATE_MAX_RETRIES   how many throttled creates to retry (default 8).

Everything describing ONE machine — the ready and build deadlines, the
delete-after and idle-hibernation backstops, the name prefix — is documented
and read in ``tb2_sandbox_boxd``, and the per-exec timeout inside it
(TB2_COMMAND_TIMEOUT_S) in ``tb2_sandbox_recipe``, next to the code each one
steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_boxd

logger = logging.getLogger(__name__)


# boxd answers a create it cannot seat yet (the org's machine limit, request
# rate) with RateLimitError; the shared backend caps in-flight creates per
# rollout and retries those with jittered exponential backoff (knobs:
# OPENENV_BOXD_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a machine create failed only because boxd would not seat it yet."""
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where boxd is already in sys.modules,
    # having just raised `exc`), and offline tests must not require the SDK.
    try:
        from boxd import RateLimitError

        if isinstance(exc, RateLimitError):
            return True
    except ImportError:  # pragma: no cover - only without the boxd SDK
        pass
    return common.throttle_text(exc)


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    machine_id, url = tb2_sandbox_boxd.create_task_sandbox(Path(tasks_dir) / task_id)
    return (lambda: tb2_sandbox_boxd.delete_sandbox(machine_id)), url


BACKEND = common.SandboxBackend(
    provider="boxd",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("BOXD"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
