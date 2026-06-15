"""
Agent V2: reward, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

Task-type agnostic — verifier reward is pre-computed by the Harbor
environment and stored in sample.metadata["reward"] regardless of task type.
For SWE agent training, the verifier reward is only used when the agent
actually submitted a solution.

Dynamic filter uses the general-purpose ``check_no_aborted`` from
``miles.rollout.filter_hub.dynamic_sampling_filters``.

Components:
  - reward_func: gates verifier reward by agent exit status
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - RolloutFn: InferenceRolloutFn subclass that logs agent metrics
"""

import logging
from collections import Counter

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


_REWARD_ELIGIBLE_EXIT_STATUSES = {"submitted"}


def _metadata_reward(sample: Sample) -> float:
    metadata = sample.metadata or {}
    reward = float(metadata.get("reward", 0.0) or 0.0)
    exit_status = str(metadata.get("exit_status", "")).lower()
    if exit_status not in _REWARD_ELIGIBLE_EXIT_STATUSES:
        return 0.0
    return reward


def _reward_inputs(sample: Sample) -> tuple[float, str, float]:
    metadata = sample.metadata or {}
    verifier_reward = float(metadata.get("reward", 0.0) or 0.0)
    exit_status = str(metadata.get("exit_status", ""))
    train_reward = _metadata_reward(sample)
    return verifier_reward, exit_status, train_reward


def _log_reward_summary(samples: list[Sample], train_rewards: list[float]) -> None:
    exit_statuses = [str((s.metadata or {}).get("exit_status", "")) for s in samples]
    verifier_rewards = [float((s.metadata or {}).get("reward", 0.0) or 0.0) for s in samples]
    gated_positive = sum(1 for verifier, train in zip(verifier_rewards, train_rewards, strict=True) if verifier > 0 and train == 0)
    logger.info(
        "[reward-gate] batch_size=%d exit_status_counts=%s verifier_reward_counts=%s "
        "train_reward_counts=%s gated_positive=%d",
        len(samples),
        dict(Counter(exit_statuses)),
        dict(Counter(verifier_rewards)),
        dict(Counter(train_rewards)),
        gated_positive,
    )


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Return verifier reward only for agent runs that submitted a solution.

    ``metadata["reward"]`` means the verifier passed. ``metadata["exit_status"]``
    tells whether the agent produced an actual submission. We only encourage
    submitted solutions here; statuses such as LimitsExceeded, Timeout, or
    Exception receive zero reward even if the verifier-side signal is positive.

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        rewards = [_metadata_reward(s) for s in samples]
        _log_reward_summary(samples, rewards)
        return rewards

    verifier_reward, exit_status, train_reward = _reward_inputs(samples)
    if verifier_reward != train_reward:
        logger.info(
            "[reward-gate] single verifier_reward=%s exit_status=%r train_reward=%s",
            verifier_reward,
            exit_status,
            train_reward,
        )
    return train_reward


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m.get(key, 0) for m in all_metrics]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


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
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
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
