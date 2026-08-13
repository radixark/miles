"""Daytona-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN Daytona cloud sandbox — built
from the task's OFFICIAL image plus an env server layer, deleted when the
episode ends. Full per-task image fidelity with zero shared infrastructure
(no Docker host, no long-lived env server) and zero cross-episode state
leakage.

The agent loop and training wrapper live in ``openenv_agent_function``
(sibling module) and are reused unchanged; this module only supplies its own
``run_episode`` — how an env comes into being (see the episode-wiring note
there). The image recipe lives in ``tb2_sandbox_recipe`` and its Daytona
materialization in ``tb2_sandbox_daytona``; the recipe bakes the installed
``tbench2_env`` package -- OpenEnv's Terminal-Bench-2 environment package --
into the image, so this variant needs the pinned tbench2_env install from
the README (canonical test.sh scoring and verifier-asset withholding built
into the server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR        path to a terminal-bench-2 checkout: build the
                     sandbox declaratively per episode. Daytona caches image
                     layers by definition hash, so only the first episode of a
                     task builds (~10 min); repeats start in ~1 min. No named
                     snapshots, so no org snapshot quota.
  DAYTONA_API_KEY              the Daytona API key, authenticating every
                     sandbox create/delete. Read from the worker's own
                     node-local environment; nothing forwards it. Supply it
                     via platform-injected pod env, or by exporting it in
                     the shell that starts ray on a single host.
  DAYTONA_API_KEY_FILE         fallback when DAYTONA_API_KEY is unset: path
                     of a file holding the key (default
                     ~/.config/daytona/api_key). Launchers forward this path
                     instead of the key itself, because ray runtime_env is
                     logged in plaintext. Point it at a file every node can
                     read: a dotfile, K8s Secret mount, or shared-FS path.
  OPENENV_DAYTONA_CREATE_CONCURRENCY  max in-flight sandbox creates (default 4).
  OPENENV_DAYTONA_CREATE_MAX_RETRIES  how many throttled creates to retry (default 8).

Everything describing ONE sandbox — the ready deadline, and the auto-stop /
auto-delete backstop — is documented and read in ``tb2_sandbox_daytona``, and
the per-exec timeout inside it (TB2_COMMAND_TIMEOUT_S) in
``tb2_sandbox_recipe``, next to the code each one steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_daytona

logger = logging.getLogger(__name__)


# Each episode materializes the per-task image declaratively from the Image
# definition, read off the local TB2 checkout (OPENENV_TB2_TASKS_DIR); repeat
# creates hit Daytona's build cache, and no named snapshot is involved.
#
# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1012 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Daytona rate-limits sandbox creation (ThrottlerException: Too Many Requests);
# the shared backend caps in-flight creates process-wide and retries throttled ones
# with jittered exponential backoff (knobs: OPENENV_DAYTONA_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Daytona rate-limited it.

    The SDK normalizes HTTP 429 to DaytonaRateLimitError; the text match is a
    fallback for older SDKs and server messages that only surface as text
    (e.g. "ThrottlerException: Too Many Requests").
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where daytona is already in sys.modules,
    # having just raised `exc`), and older SDKs lack the class entirely.
    try:
        from daytona.common.errors import DaytonaRateLimitError

        if isinstance(exc, DaytonaRateLimitError):
            return True
    except ImportError:  # pragma: no cover - only without the daytona SDK
        pass
    return common.throttle_text(exc, "throttler")


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    daytona = tb2_sandbox_daytona.make_daytona()
    sandbox, url = tb2_sandbox_daytona.create_task_sandbox(daytona, Path(tasks_dir) / task_id)
    return (lambda: daytona.delete(sandbox)), url


BACKEND = common.SandboxBackend(
    provider="Daytona",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("DAYTONA"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
