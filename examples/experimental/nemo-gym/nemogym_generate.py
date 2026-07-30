"""NeMo-Gym example: reward hook.

The generate function is provided by
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to nemogym_agent_function.run.

Reward is pre-computed by the NeMo-Gym environment (SWE-bench harness) during
the episode and stored in sample.metadata["reward"].
"""

from miles.utils.types import Sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the NeMo-Gym environment during generate()."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
