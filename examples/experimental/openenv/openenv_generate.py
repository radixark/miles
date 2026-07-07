"""Reward function for the OpenEnv Phase-1 smoke run.

Task-agnostic: the agent function (``openenv_echo_agent_function.run``) stores the
env-provided reward in ``sample.metadata["reward"]``; this just reads it back.
Mirrors ``swe-agent-v2/generate.py:reward_func`` so it works for both the
single-sample (``async_rm``) and batched (``--custom-rm-path``) call paths.
"""

from miles.utils.types import Sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
