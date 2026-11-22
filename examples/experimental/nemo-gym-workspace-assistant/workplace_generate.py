"""Validate Workplace episodes before they can enter an optimizer batch."""

import math
from typing import Any

import numpy as np

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as agentic_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    output = await agentic_generate(input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    for sample in samples:
        if not sample.metadata.get("workplace_episode_valid"):
            sample.status = Sample.Status.ABORTED
            continue
        if sample.status == Sample.Status.ABORTED:
            continue
        sample.validate()
        if sample.loss_mask is None or not any(sample.loss_mask):
            raise ValueError("Workplace episode has no policy tokens to train")
        if sample.metadata["workplace_turns"] > 1 and all(sample.loss_mask):
            raise ValueError("Multi-turn tool observations must be masked out")
        if sample.rollout_log_probs is None or not all(math.isfinite(p) for p in sample.rollout_log_probs):
            raise ValueError("Missing or nonfinite rollout log probabilities")
        routes = sample.rollout_routed_experts
        if input.args.use_rollout_routing_replay:
            if routes is None or routes.shape != (
                len(sample.tokens) - 1,
                input.args.num_layers,
                input.args.moe_router_topk,
            ):
                raise ValueError("Missing or misaligned routing replay")
            if routes.dtype != np.int32 or routes.min() < 0 or routes.max() >= input.args.num_experts:
                raise ValueError("Invalid routing replay expert indices")
        if sample.metadata["workplace_stop"] in {"truncated", "response_budget", "context_limit", "turn_limit"}:
            sample.status = Sample.Status.TRUNCATED
    return output


generate.add_arguments = agentic_generate.add_arguments


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if not sample.metadata.get("workplace_episode_valid"):
        raise ValueError("An invalid episode reached the reward hook")
    value = sample.metadata["workplace_reward"]
    if value not in (0.0, 1.0):
        raise ValueError(f"Unexpected native Workplace reward: {value!r}")
    return float(value)
