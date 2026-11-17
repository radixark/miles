"""tau2-bench generate wrapper and reward hook.

``generate`` is ``miles.rollout.generate_hub.agentic_tool_call.generate`` with the
rollout router's OpenAI endpoint added to the sample metadata, so the tau2 user
simulator can run on the policy engines without going through the traced session,
and with an invalid episode (``tau2_episode_valid`` False: the simulation raised, or the
agent process never reported) marked ``ABORTED`` so its group is dropped; the generic
generate would keep it as a scored sample. See the README's "Episode outcomes".

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
    output = await agentic_tool_call.generate(input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    for sample in samples:
        if sample.status == Sample.Status.ABORTED:
            continue
        if not sample.metadata.get("tau2_episode_valid"):
            sample.status = Sample.Status.ABORTED
    return output


generate.add_arguments = agentic_tool_call.generate.add_arguments


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
