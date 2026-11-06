"""tau2-bench <-> miles adapter (agent function).

miles calls ``run`` once per sample via
``--custom-agent-function-path tau2_agent_function.run`` (with
``--custom-generate-function-path tau2_generate.generate``). Each call runs one
unmodified tau2 text simulation (``tau2.run.run_single_task``): tau2's own
``llm_agent`` talks to the policy through miles' session server, so every agent
turn is recorded as token ids + logprobs + loss masks, while tau2 keeps owning
the user simulator, the user-side tools (telecom), the stop conditions and the
evaluator. The evaluator's reward is returned and lands in ``sample.metadata``.

The user simulator and the NL-assertion judge (retail rewards are DB x
NL_ASSERTION) are reached at ``metadata["tau2_user_api_base"]``, which
``tau2_generate.generate`` fills with the rollout router: the policy model also
plays the user and the judge, with thinking off and without being traced. Set
``TAU2_USER_LLM`` / ``TAU2_JUDGE_LLM`` (a litellm model id such as ``gpt-4.1``) plus
that provider's key to use an external model instead.

Env vars:
  TAU2_USER_LLM            user-simulator litellm model (default: the rollout router)
  TAU2_USER_API_BASE       user-simulator endpoint override
  TAU2_JUDGE_LLM           NL-assertion judge litellm model (default: the rollout router)
  TAU2_JUDGE_API_BASE      judge endpoint override
  TAU2_COMMUNICATE_SCORER  substring (default, tau2's original matcher) | semantic
                           (tau2 >= 2.0 default: an OpenAI judge, needs OPENAI_API_KEY)
  TAU2_MAX_STEPS           tau2 orchestrator step cap (default: 100)
  TAU2_EPISODE_TIMEOUT_S   wall-clock cap for one episode (default: 1800)
"""

import asyncio
import logging
import os
import uuid
from typing import Any

from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_single_task

from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)

_POLICY_MODEL = "openai/policy"


def _side_llm(
    metadata: dict[str, Any], external_llm: str | None, api_base: str | None, **local_args
) -> tuple[str, dict[str, Any]]:
    if external_llm:
        llm_args: dict[str, Any] = {"temperature": 0.0}
        if api_base:
            llm_args["api_base"] = api_base
        return external_llm, llm_args
    return _POLICY_MODEL, {
        "api_base": api_base or metadata["tau2_user_api_base"],
        "api_key": "EMPTY",
        "temperature": 0.0,
        "extra_headers": {"X-SMG-Routing-Key": f"tau2-side-{uuid.uuid4().hex}"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        **local_args,
    }


def _run_simulation(
    domain: str, task_split: str, task_id: str, agent_llm_args: dict, user: tuple, judge: tuple
) -> SimulationRun:
    # one episode per process (subproc mode), so rebinding tau2's judge constants is local to it
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS, evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge

    (task,) = get_tasks(domain, task_split_name=task_split, task_ids=[task_id])
    llm_user, llm_args_user = user
    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=_POLICY_MODEL,
        llm_args_agent=agent_llm_args,
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        max_steps=int(os.getenv("TAU2_MAX_STEPS", "100")),
        communicate_scoring=CommunicateScoring(
            scorer=CommunicateScorer(os.getenv("TAU2_COMMUNICATE_SCORER", CommunicateScorer.SUBSTRING.value))
        ),
    )
    return run_single_task(config, task, evaluation_type=EvaluationType.ALL)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    domain, task_split, task_id = metadata["domain"], metadata["task_split"], metadata["task_id"]
    agent_llm_args = {
        **(request_kwargs or {}),
        "api_base": openai_session_url(base_url),
        "api_key": "EMPTY",
        "num_retries": 0,
    }
    timeout_s = float(os.getenv("TAU2_EPISODE_TIMEOUT_S", "1800"))
    result = {"reward": 0.0, "tau2_domain": domain, "tau2_task_id": task_id}
    try:
        simulation = await asyncio.wait_for(
            asyncio.to_thread(
                _run_simulation,
                domain,
                task_split,
                task_id,
                agent_llm_args,
                _side_llm(metadata, os.getenv("TAU2_USER_LLM"), os.getenv("TAU2_USER_API_BASE")),
                _side_llm(
                    metadata,
                    os.getenv("TAU2_JUDGE_LLM"),
                    os.getenv("TAU2_JUDGE_API_BASE"),
                    response_format={"type": "json_object"},
                ),
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error(f"tau2 {domain}/{task_id} timed out after {timeout_s:.0f}s")
        return {**result, "tau2_termination_reason": "episode_timeout"}

    reward = simulation.reward_info.reward if simulation.reward_info is not None else 0.0
    return {
        **result,
        "reward": float(reward),
        "tau2_termination_reason": simulation.termination_reason.value,
        "tau2_num_messages": len(simulation.messages),
    }
