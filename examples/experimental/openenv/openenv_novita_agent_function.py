"""Novita-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN sandbox on
`Novita Agent Sandbox <https://docs.novita.ai>`_ — built from the task's
OFFICIAL image plus an env server layer, killed when the episode ends.

The agent loop and training wrapper live in ``openenv_agent_function``, and
everything about running episodes on per-episode sandboxes is
``openenv_sandbox_common``'s ``SandboxBackend``. This module is only what is
Novita-specific: how one sandbox comes into being, which of its SDK's errors
are worth retrying, and its knobs. The image recipe lives in
``tb2_sandbox_recipe`` and its Novita materialization in
``tb2_sandbox_novita``; like the other sandbox backends, this variant needs the
pinned tbench2_env install from the README (canonical test.sh scoring and
verifier-asset withholding built into the server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. Templates
                    are built once per task under a recipe-digest name
                    (first episode of a task pays the build; repeats start
                    from the template), or pre-baked via the
                    tb2_sandbox_novita CLI.
  NOVITA_API_KEY / NOVITA_API_KEY_FILE   key supply on the contract every
                    backend shares (file default ~/.config/novita/api_key;
                    launchers forward the PATH, never the value).
  NOVITA_DOMAIN               region, read by the SDK itself (unset: us-phx-1,
                    the v2 region). NOVITA_API_URL / NOVITA_SANDBOX_URL
                    override the endpoints for an internal deployment.
  OPENENV_NOVITA_CREATE_CONCURRENCY   max in-flight sandbox creates (default 4).
  OPENENV_NOVITA_CREATE_MAX_RETRIES   how many throttled creates to retry (default 8).

Everything describing ONE sandbox — its lifetime and the ready deadline — is
documented and read in ``tb2_sandbox_novita``, and the per-exec timeout inside
it (TB2_COMMAND_TIMEOUT_S) in ``tb2_sandbox_recipe``, next to the code each one
steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_novita

logger = logging.getLogger(__name__)


# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1025 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Novita rate-limits API calls and caps an account's concurrent sandboxes by
# plan; the shared backend caps in-flight creates per rollout and retries
# throttled ones with jittered exponential backoff (knobs: OPENENV_NOVITA_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because Novita would not seat it yet.

    The SDK types HTTP 429 as RateLimitException; the text match also catches
    the account's concurrent-sandbox quota, which frees up the same way.
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where novita_sandbox is already in
    # sys.modules, having just raised `exc`), and offline tests must not
    # require the SDK.
    try:
        from novita_sandbox import RateLimitException

        if isinstance(exc, RateLimitException):
            return True
    except ImportError:  # pragma: no cover - only without the novita SDK
        pass
    return common.throttle_text(exc, "quota")


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    sandbox, url = tb2_sandbox_novita.create_task_sandbox(Path(tasks_dir) / task_id)
    return (lambda: tb2_sandbox_novita.kill_sandbox(sandbox)), url


BACKEND = common.SandboxBackend(
    provider="Novita",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("NOVITA"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
