"""E2B-sandbox variant of the OpenEnv tbench2 agent function.

A drop-in alternative to ``openenv_agent_function.run`` for
``--custom-agent-function-path``: instead of sharing one env server
(OPENENV_ENV_URL), every episode gets its OWN sandbox on an E2B-compatible
provider — built from the task's OFFICIAL image plus an env server layer,
killed when the episode ends. The provider is whatever the E2B SDK is pointed
at: E2B Cloud, or a self-hosted AgentENV deployment
(https://github.com/kvcache-ai/AgentENV) via E2B_API_URL / E2B_SANDBOX_URL.

The agent loop and training wrapper live in ``openenv_agent_function``
(sibling module) and are reused unchanged; this module only supplies its own
``run_episode`` — how an env comes into being (see the episode-wiring note
there). The image recipe lives in ``tb2_sandbox_recipe`` and its E2B
materialization in ``tb2_sandbox_e2b``; like every sandbox backend, this one
needs the pinned tbench2_env install from the README (canonical test.sh
scoring and verifier-asset withholding built into the server).

Env vars (the agent-loop ones in ``openenv_agent_function`` apply too):
  OPENENV_TB2_TASKS_DIR       path to a terminal-bench-2 checkout. Templates
                    are built once per task under a recipe-digest alias
                    (first episode of a task pays the build; repeats
                    warm-start), or pre-baked via the tb2_sandbox_e2b CLI.
  E2B_API_KEY / E2B_API_KEY_FILE   key supply on the contract every backend
                    shares (file default ~/.config/e2b/api_key; launchers
                    forward the PATH, never the value). AgentENV accepts any
                    non-empty key today.
  E2B_API_URL / E2B_SANDBOX_URL    endpoint overrides read by the SDK itself;
                    set both to target a self-hosted AgentENV.
  OPENENV_E2B_CREATE_CONCURRENCY   max in-flight sandbox creates (default 4).
  OPENENV_E2B_CREATE_MAX_RETRIES   how many throttled creates to retry (default 8).
  OPENENV_E2B_THROTTLE_PATTERNS    extra comma-separated lowercase substrings
                    classified as retryable throttling/capacity errors — a
                    self-hosted AgentENV words "at capacity" its own way, and
                    whoever deployed it can name that without a code change.

Everything describing ONE sandbox — its lifetime and the ready deadline — is
documented and read in ``tb2_sandbox_e2b``, and the per-exec timeout inside it
(TB2_COMMAND_TIMEOUT_S) in ``tb2_sandbox_recipe``, next to the code each one
steers.
"""

import logging
from pathlib import Path
from typing import Any

import openenv_sandbox_common as common
import tb2_sandbox_e2b

logger = logging.getLogger(__name__)


# The sandbox's env server is the tbench2_env baked by the recipe, installed
# per the README (at or after the huggingface/OpenEnv#1012 merge): canonical
# tests/test.sh scoring built into `evaluate`, task WORKDIR resolved
# server-side, verifier assets withheld. The launcher preflight rejects an
# older install outright, and the shared agent loop's harness-marker guard
# backstops it per episode.
#
# Providers rate-limit or run out of capacity under a fanned-out rollout; the
# shared backend caps in-flight creates process-wide and retries throttled ones
# with jittered exponential backoff (knobs: OPENENV_E2B_CREATE_*).
def _is_throttle_error(exc: BaseException) -> bool:
    """True when a sandbox create failed only because the provider throttled it.

    The SDK types HTTP 429 as RateLimitException; the text match covers
    proxies and self-hosted servers whose limits only surface as text, and
    OPENENV_E2B_THROTTLE_PATTERNS extends the set for provider-specific
    capacity errors (e.g. a full AgentENV node pool) without a code change.
    """
    # In-function import, deliberately: this is the class's only use site, it
    # only runs on the failure path (where e2b is already in sys.modules,
    # having just raised `exc`), and offline tests must not require the SDK.
    try:
        from e2b.exceptions import RateLimitException

        if isinstance(exc, RateLimitException):
            return True
    except ImportError:  # pragma: no cover - only without the e2b SDK
        pass
    return common.throttle_text(exc, patterns_env_var="OPENENV_E2B_THROTTLE_PATTERNS")


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Any, str]:
    sandbox, url = tb2_sandbox_e2b.create_task_sandbox(Path(tasks_dir) / task_id)
    return (lambda: tb2_sandbox_e2b.kill_sandbox(sandbox)), url


BACKEND = common.SandboxBackend(
    provider="E2B",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("E2B"),
)

# Module-level entry points: `--custom-agent-function-path` resolves a module
# attribute, and the sibling tools reach run_episode the same way.
run_episode = BACKEND.run_episode
run = BACKEND.run
