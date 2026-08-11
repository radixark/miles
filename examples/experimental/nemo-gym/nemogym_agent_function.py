"""NeMo-Gym <-> miles adapter (agent function).

Targets upstream NVIDIA-NeMo/Gym's sandbox-backed ``mini_swe_agent_2`` agent,
which requires the per-request policy endpoint override (upstream ``main``,
>= ``fcca3a8``).

miles calls ``run`` once per sample via
``--custom-agent-function-path nemogym_agent_function.run`` (with
``--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate``).
Each call POSTs the task to the NeMo-Gym agent server's ``/run`` endpoint,
handing over the session's OpenAI-compatible URL as ``policy_base_url``.
NeMo-Gym runs mini-swe-agent v2 in a ``nemo_gym.sandbox`` container (docker /
daytona / apptainer / ecs_fargate / opensandbox providers) against that URL,
so every model call goes through miles' session server and is recorded
losslessly (token ids + logprobs + loss masks) — no re-tokenization.

The NeMo-Gym environment grades the episode itself (SWE-bench harness); the
returned dict is merged into ``sample.metadata`` so the reward hook
(``--custom-rm-path nemogym_generate.reward_func``) can read
``metadata["reward"]``.

Env vars:
  NEMO_GYM_URL   base URL of the NeMo-Gym agent server
                 (default: http://localhost:12000).
  NEMO_GYM_RUN_TIMEOUT  hard wall-clock cap in seconds for one /run call
                 (default: 3600). SWE episodes pull per-task docker images on
                 first use, which can dominate early rollouts.
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for the session URL when
                 the NeMo-Gym server cannot resolve the trainer's hostname
                 (e.g. it runs outside the trainer's docker network).
"""

import asyncio
import logging
import os
import random
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

# Deliberately no miles imports: importing miles pulls in torch, and this
# adapter (plus its offline tests and the eval_nemogym_via_api scan) must load
# on CPU-only machines too.

_POST_ATTEMPTS = 3
_POST_BACKOFF_S = (1.0, 5.0)


async def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON and return the JSON response, retrying transport errors.

    No HTTP timeout here: the caller bounds the whole episode with
    asyncio.wait_for (a /run legitimately takes minutes).
    """
    async with httpx.AsyncClient(timeout=None) as client:
        for attempt in range(_POST_ATTEMPTS):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                if attempt == _POST_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(random.uniform(*_POST_BACKOFF_S))


def _resolve_session_url(base_url: str) -> str:
    """Build the OpenAI-compatible policy URL, rewriting host for off-cluster agents."""
    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        netloc = f"{external_host}:{parsed.port}" if parsed.port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))
    return session_url


def build_responses_create_params(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map miles' chat-completions sampling kwargs onto NeMo-Gym's Responses-API params.

    mini_swe_agent_2 reads sampling settings exclusively from
    ``responses_create_params`` (temperature / top_p / max_output_tokens, see
    upstream ``_responses_create_params_to_model_kwargs``).
    """
    params: dict[str, Any] = {"input": []}
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    if request_kwargs.get("max_tokens") is not None:
        params["max_output_tokens"] = request_kwargs["max_tokens"]
    return params


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one task instance via the NeMo-Gym mini_swe_agent_2 server.

    Returns the reward dict to merge into sample metadata, or None on a
    transport failure (timeout, unreachable server). On None the recorded
    session still becomes a sample; the reward hook then scores it 0.0 via its
    default. Episodes that never reach the model produce no session records
    and are ABORTED by the generate layer, so
    ``--dynamic-sampling-filter-path .. check_no_aborted`` drops their group.
    """
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    nemo_gym_url = os.getenv("NEMO_GYM_URL", "http://localhost:12000")
    timeout_s = float(os.getenv("NEMO_GYM_RUN_TIMEOUT", "3600"))

    # The SWE-bench-format instance fields (instance_id, repo, base_commit,
    # problem_statement, subset, split, ...) ride in metadata straight from the
    # prompt data and must sit at the top level of the run body: the server
    # uses body.model_dump() as the instance dict (image selection, eval).
    request: dict[str, Any] = {
        **metadata,
        "responses_create_params": build_responses_create_params(request_kwargs),
        "policy_base_url": _resolve_session_url(base_url),
    }

    try:
        response = await asyncio.wait_for(
            post_json(f"{nemo_gym_url}/run", request),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error(f"NeMo-Gym /run timed out after {timeout_s:.0f}s")
        return None
    except asyncio.CancelledError:
        logger.warning("NeMo-Gym /run cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"NeMo-Gym /run failed: {e}")
        return None

    return {
        "reward": response.get("reward", 0.0),
        # The SWE-bench eval report (tests_status, patch_successfully_applied,
        # or {"error": ...} when the episode failed server-side) rides in the
        # response's `metadata`.
        "eval_report": response.get("metadata", {}) or {},
    }
