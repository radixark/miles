"""tau2-bench generate wrapper and reward hook.

``generate`` is ``miles.rollout.generate_hub.agentic_tool_call.generate`` with the
rollout router's OpenAI endpoint added to the sample metadata, so the tau2 user
simulator can run on the policy engines without going through the traced session,
and with the sample status set from how ``tau2_agent_function.run`` says the
episode ended (the generic generate keeps a failed episode as a scored sample):

- ``tau2_episode_valid`` False (the episode raised, or the agent process never
  reported): ``ABORTED``, so the group is dropped;
- ``tau2_termination_reason == "max_seq_len"`` (the session refused the next turn,
  HTTP 409 from ``--max-seq-len``): ``TRUNCATED`` with reward 0, which
  ``--rollout-sample-filter-path mask_truncated_zero_reward`` leaves out of the loss;
- otherwise the status the session assembled.

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
        elif sample.metadata.get("tau2_termination_reason") == "max_seq_len":
            sample.status = Sample.Status.TRUNCATED
    return output


generate.add_arguments = agentic_tool_call.generate.add_arguments


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
