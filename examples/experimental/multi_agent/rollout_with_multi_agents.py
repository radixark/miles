import random
from copy import deepcopy
from typing import Any

from examples.experimental.multi_agent.agent_system import AgentState
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import load_function
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.experimental.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(
    args: ImmutableNamespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:

    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    agent_args = ImmutableNamespace.model_validate(
        dict(args) | {"rollout_max_context_len": max_context_length, **MULTI_AGENT_CONFIGS}
    )
    state = AgentState(sample=sample, tokenizer=tokenizer, sampling_params=deepcopy(sampling_params))

    custom_multi_agent_func = load_function(agent_args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(agent_args, state=state)

    random.shuffle(samples)

    return samples
