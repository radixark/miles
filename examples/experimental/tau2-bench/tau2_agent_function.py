"""τ²-bench (tau2) <-> miles adapter (agent function).

miles calls ``run`` once per sample via
``--custom-agent-function-path tau2_agent_function.run`` (with
``--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate``).
Each call plays one tau2 task with tau2's own half-duplex orchestrator, user
simulator, environment and evaluator. Only the agent is replaced: ``SessionAgent``
keeps tau2's ``LLMAgent`` system prompt and tools, but sends every turn to the
session's OpenAI-compatible endpoint (``base_url/v1``). The session server records
each policy turn's token ids, logprobs and loss mask, so the trajectory is trained
on exactly what was sampled, with no re-tokenization.

The sample metadata names the task, ``{"domain": "retail", "task_id": "0"}``;
``run.py`` writes it from a tau2 split.

Reward is tau2's own evaluation of the finished simulation (``run_simulation``:
DB / environment assertions, plus the action, communicate and NL-assertion checks
the task's reward basis lists). The returned dict lands in ``sample.metadata``,
where ``reward_func`` (``--custom-rm-path tau2_agent_function.reward_func``) reads
it back:

  - A finished simulation keeps tau2's reward, which is already 0 for every
    termination other than an agent or user stop (max steps, too many errors).
  - An episode the policy ends early scores 0: a reply cut at ``max_tokens``, a
    trajectory reaching ``--max-seq-len``, or a malformed or empty reply.
  - An episode that fails for any other reason (the user simulator or a judge
    API, the session server) has no reward. ``reward_func`` returns None, and
    training drops the group rather than learn from a false 0.

Env vars (read by every episode):
  TAU2_USER_LLM        litellm model of the user simulator (default: tau2's
                       ``DEFAULT_LLM_USER``).
  TAU2_USER_LLM_ARGS   JSON object of litellm kwargs for it (default: tau2's
                       ``DEFAULT_LLM_ARGS_USER``), e.g. the ``api_base`` of a
                       self-hosted model.
  TAU2_JUDGE_LLM       litellm model for both of tau2's reward judges, the
                       NL-assertion judge and the semantic communicate judge
                       (default: tau2's own judge for each).
  TAU2_JUDGE_LLM_ARGS  JSON object of litellm kwargs for it (default: none).
"""

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import ValidationError
from tau2.agent.llm_agent import LLMAgent
from tau2.config import (
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, SystemMessage, ToolCall
from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import RewardType
from tau2.evaluator import evaluator_nl_assertions
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.build import agent_domain_policy, build_environment, build_user
from tau2.runner.helpers import get_tasks
from tau2.runner.simulation import run_simulation
from tau2.utils.llm_utils import to_litellm_messages

from miles.rollout.agentic.session import openai_session_url

if TYPE_CHECKING:
    # Annotation only: miles.utils.types imports torch, about a second in every episode's process.
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# The session server serves one policy; the name only satisfies the OpenAI schema.
_POLICY_MODEL = "default"


class PolicyEnded(Exception):
    """The policy ended the episode before tau2 could; any such early end scores 0."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class SessionAgent(LLMAgent):
    """tau2's ``LLMAgent`` (same system prompt and tools) served by the miles session endpoint.

    The agent state is the conversation in OpenAI wire form. Policy replies stay exactly as the
    session server returned them: it matches every later request against its stored turns, so a
    reply re-serialized through tau2's message types would no longer match the recorded prefix.
    """

    def __init__(
        self,
        tools: list,
        domain_policy: str,
        *,
        client: OpenAI,
        request_kwargs: dict[str, Any],
        max_seq_len: int | None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=_POLICY_MODEL)
        self.client = client
        self.request_kwargs = request_kwargs
        self.max_seq_len = max_seq_len
        self.tools_schema = [tool.openai_schema for tool in tools]
        self.generation_time = 0.0

    def get_init_state(self, message_history: list[Message] | None = None) -> list[dict[str, Any]]:
        return _to_wire([SystemMessage(role="system", content=self.system_prompt), *(message_history or [])])

    def generate_next_message(
        self, message: Message, state: list[dict[str, Any]]
    ) -> tuple[AssistantMessage, list[dict[str, Any]]]:
        state.extend(_to_wire(message.tool_messages if isinstance(message, MultiToolMessage) else [message]))
        started = time.monotonic()
        completion = self.client.chat.completions.create(
            model=_POLICY_MODEL, messages=state, tools=self.tools_schema, extra_body=self.request_kwargs
        )
        self.generation_time += time.monotonic() - started
        choice = completion.choices[0]
        state.append(choice.message.model_dump(exclude_none=True))
        if choice.finish_reason == "length":
            raise PolicyEnded("length")
        if self.max_seq_len is not None and completion.usage.total_tokens >= self.max_seq_len:
            raise PolicyEnded(TerminationReason.CONTEXT_WINDOW_EXCEEDED.value)
        return _to_tau2_reply(choice.message), state


def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """tau2 messages in OpenAI wire form, by tau2's own conversion, without null fields."""
    return [{key: value for key, value in m.items() if value is not None} for m in to_litellm_messages(messages)]


def _to_tau2_reply(message: ChatCompletionMessage) -> AssistantMessage:
    try:
        tool_calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=json.loads(call.function.arguments))
            for call in message.tool_calls or []
        ]
    except (json.JSONDecodeError, ValidationError) as e:
        raise PolicyEnded(TerminationReason.AGENT_ERROR.value) from e
    reply = AssistantMessage(role="assistant", content=message.content, tool_calls=tool_calls or None)
    if not (reply.has_text_content() or reply.is_tool_call()):
        raise PolicyEnded(TerminationReason.AGENT_ERROR.value)
    return reply


def _user_llm() -> tuple[str, dict[str, Any]]:
    """The user simulator's litellm model and kwargs: ``TAU2_USER_LLM*``, else tau2's defaults."""
    if "TAU2_USER_LLM_ARGS" in os.environ:
        model_args = json.loads(os.environ["TAU2_USER_LLM_ARGS"])
    else:
        model_args = dict(DEFAULT_LLM_ARGS_USER)
    return os.environ.get("TAU2_USER_LLM", DEFAULT_LLM_USER), model_args


def _judge_llm() -> tuple[str, dict[str, Any]] | None:
    """The litellm model and kwargs of every tau2 judge under ``TAU2_JUDGE_LLM``; None keeps tau2's judges."""
    if "TAU2_JUDGE_LLM" not in os.environ:
        return None
    return os.environ["TAU2_JUDGE_LLM"], json.loads(os.environ.get("TAU2_JUDGE_LLM_ARGS", "{}"))


def episode_llms(reward_bases: set[RewardType]) -> list[tuple[str, dict[str, Any]]]:
    """Each litellm model, with its kwargs, that an episode on tasks with these reward bases calls."""
    judge = _judge_llm()
    llms = [_user_llm()]
    if RewardType.NL_ASSERTION in reward_bases:
        llms.append(
            judge
            or (
                evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS,
                evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
            )
        )
    if RewardType.COMMUNICATE in reward_bases:
        default = CommunicateScoring().judge
        llms.append(judge or (default.model, default.model_args))
    return llms


def _configure_judges() -> CommunicateScoring:
    """Point tau2's judges at ``TAU2_JUDGE_LLM`` when it is set; returns the communicate scoring to evaluate with."""
    judge = _judge_llm()
    if judge is None:
        return CommunicateScoring()
    model, model_args = judge
    # tau2 has no parameter for the NL-assertion judge: its evaluator reads these constants at each call.
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = model
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = model_args
    return CommunicateScoring(judge=SemanticCommunicateSettings(model=model, model_args=model_args))


def _play_task(
    session_url: str, domain: str, task_id: str, request_kwargs: dict[str, Any], max_seq_len: int | None
) -> dict[str, Any]:
    """Play one tau2 task against the policy at ``session_url``; blocking."""
    (task,) = get_tasks(domain, task_ids=[task_id])
    environment = build_environment(domain)
    user_llm, user_llm_args = _user_llm()
    user = build_user(DEFAULT_USER_IMPLEMENTATION, environment, task, llm=user_llm, llm_args=user_llm_args)
    communicate_scoring = _configure_judges()
    started = time.monotonic()
    with OpenAI(base_url=session_url, api_key="EMPTY") as client:
        agent = SessionAgent(
            tools=environment.get_tools(),
            domain_policy=agent_domain_policy(environment),
            client=client,
            request_kwargs=request_kwargs,
            max_seq_len=max_seq_len,
        )
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=DEFAULT_MAX_STEPS,
            max_errors=DEFAULT_MAX_ERRORS,
        )
        try:
            simulation = run_simulation(orchestrator, communicate_scoring=communicate_scoring)
            reward, termination_reason = simulation.reward_info.reward, simulation.termination_reason.value
        except PolicyEnded as e:
            reward, termination_reason = 0.0, e.reason
        except Exception as e:
            logger.error(f"tau2 {domain}/{task_id} failed outside the policy: {e!r}", exc_info=True)
            reward, termination_reason = None, TerminationReason.INFRASTRUCTURE_ERROR.value
    return {
        "reward": reward,
        "termination_reason": termination_reason,
        # User simulator, tool and judge time, which throughput accounting leaves out.
        "agent_metrics": {"total_tool_time": time.monotonic() - started - agent.generation_time},
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Play the tau2 task the sample metadata names; tau2's orchestrator is synchronous, so it runs in a thread."""
    metadata = metadata or {}
    return await asyncio.to_thread(
        _play_task,
        openai_session_url(base_url),
        metadata["domain"],
        str(metadata["task_id"]),
        request_kwargs or {},
        metadata.get("max_seq_len"),
    )


async def reward_func(args, samples: "Sample | list[Sample]", **kwargs) -> float | None | list[float | None]:
    """The reward ``run`` left in metadata; None marks an episode tau2 could not score."""
    if isinstance(samples, list):
        return [sample.metadata.get("reward") for sample in samples]
    return samples.metadata.get("reward")
