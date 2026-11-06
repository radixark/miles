"""tau2-bench generate wrapper and reward hook.

``generate`` is ``miles.rollout.generate_hub.agentic_tool_call.generate`` with the
rollout router's OpenAI endpoint added to the sample metadata, so the tau2 user
simulator can run on the policy engines without going through the traced session.

The reward is computed by the tau2 evaluator inside the episode and stored in
``sample.metadata["reward"]`` by ``tau2_agent_function.run``.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    input.sample.metadata["tau2_user_api_base"] = (
        f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}/v1"
    )
    return await agentic_tool_call.generate(input)


generate.add_arguments = agentic_tool_call.generate.add_arguments


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
