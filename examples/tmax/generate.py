"""TMax reward function and rollout wrapper.

The reward is computed in the sandbox by the task's tests (tmax_agent_function.run)
and rides back in ``sample.metadata["reward"]``. ``None`` there means the sandbox
failed for a non-policy reason; returning it lets the missing-reward filter drop
the group instead of training on a false 0.
"""

import logging
from collections import Counter

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MEAN_KEYS = ("turns", "tool_calls", "submitted", "model_query_time_sum", "env_execution_time_sum", "total_time")


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return [s.metadata.get("reward") for s in samples]
    return samples.metadata.get("reward")


def aggregate_agent_metrics(samples: list[Sample]) -> dict[str, float]:
    all_metrics = [s.metadata["agent_metrics"] for s in samples if (s.metadata or {}).get("agent_metrics")]
    if not all_metrics:
        return {}
    metrics = {
        f"agent/{key}_mean": sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in _MEAN_KEYS
        if all(key in m for m in all_metrics)
    }
    for reason, count in Counter(m["end_reason"] for m in all_metrics).items():
        metrics[f"agent/end_reason/{reason}"] = count / len(all_metrics)
    return metrics


class RolloutFn(InferenceRolloutFn):
    """InferenceRolloutFn that also logs agent/* metrics for the kept samples."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)
        # A group holds a Sample per episode, or with session server v2 a list[Sample] per episode.
        samples = [s for group in output.samples for s in iter_samples(group)]
        agent_metrics = aggregate_agent_metrics(samples)
        if agent_metrics:
            output.metrics = {**(output.metrics or {}), **agent_metrics}
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")
        return output
