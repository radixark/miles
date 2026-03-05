"""
SWE-Agent V2: reward, filter, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

This file only contains SWE-Agent-specific components that sit outside
the generic agentic generate loop:
  - reward_func: reads pre-computed reward from sample metadata
  - dynamic_filter: rejects groups with any aborted sample
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - RolloutFn: InferenceRolloutFn subclass that logs agent metrics
"""

import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the SWE-Agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Dynamic Filter --


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject entire group if any sample is aborted."""
    flat = samples if not isinstance(samples[0], list) else [s for group in samples for s in group]
    if any(s.status == Sample.Status.ABORTED for s in flat):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


# -- Agent Metrics Aggregation --


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    for key in ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    for key in ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    for key in ["model_time_ratio", "env_time_ratio", "eval_time_ratio"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    values = [m.get("total_time", 0) for m in all_metrics]
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
