import asyncio
import logging
import time
import uuid
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent

from minisweagent.environments import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.swegym_runner import get_swegym_docker_image_name, run_eval

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.sglang_rollout import GenerateState, eval_rollout
from miles.utils.async_utils import run
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:

    if not messages or len(messages) < 2:
        return [], [], "", 0

    prompt_msgs = messages[:2]
    response_msgs = messages[2:]

    prompt_tokens = []
    for msg in prompt_msgs:
        content = msg.get("content", "")
        if content:
            prompt_tokens.extend(tokenizer(content, add_special_tokens=False)["input_ids"])

    response_tokens = []
    loss_mask = []
    response_text_parts = []

    for msg in response_msgs:
        content = msg.get("content", "")
        if not content:
            continue

        tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        token_len = len(tokens)

        response_tokens.extend(tokens)
        response_text_parts.append(content)

        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * token_len)

    all_tokens = prompt_tokens + response_tokens
    response_text = "".join(response_text_parts)
    response_length = len(response_tokens)

    return all_tokens, loss_mask, response_text, response_length


def run_agent_sync_logic(model, env, problem_statement, sampling_params, metadata, instance_dir, run_id):
    """
    Synchronous wrapper to run the agent and evaluation.
    This is offloaded to a thread to prevent blocking the Ray actor.
    """
    agent = DefaultAgent(
        model=model,
        env=env,
        responses_create_params={"input": []},
        sampling_params=sampling_params,
        step_limit=250,
        collapse_limit=3,
    )

    # Execute the agent lifecycle
    exit_status, result_patch, agent_metrics = agent.run(problem_statement)

    # Run evaluation
    eval_start = time.time()
    eval_report_full = run_eval(
        instance=metadata,
        env=env,
        model_patch=result_patch,
        instance_dir=instance_dir,
        run_id=run_id,
    )
    eval_time = time.time() - eval_start

    # metrics calculation
    agent_metrics["eval_time"] = eval_time
    total_time = agent_metrics.get("agent_run_time", 0) + eval_time
    agent_metrics["total_time"] = total_time
    agent_metrics["model_time_ratio"] = agent_metrics.get("model_query_time_sum", 0) / max(total_time, 1e-6)
    agent_metrics["env_time_ratio"] = agent_metrics.get("env_execution_time_sum", 0) / max(total_time, 1e-6)
    agent_metrics["eval_time_ratio"] = eval_time / max(total_time, 1e-6)

    return exit_status, agent.messages, eval_report_full, agent_metrics


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Custom generation function for SWE-Agent integration.

    Orchestrates the interaction with the external Gym environment:
    1. Directly initializes mini-swe-agent components.
    2. Runs agent logic in a background thread to maintain Ray cluster stability.
    3. Formats data for Miles training format.

    Note: Performs in-place modification of `sample` for memory efficiency.
    """
    instance_id = sample.metadata.get("instance_id")
    subset = sample.metadata.get("subset", "gym")
    problem_statement = sample.metadata.get("problem_statement")

    # Model configuration
    model_name = f"sglang/{Path(args.hf_checkpoint).name}"
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"

    model_config = {"model_name": model_name, "model_kwargs": {"base_url": sglang_url, "api_key": "dummy"}}
    model = get_model(model_name, config=model_config)

    # Environment configuration
    image_name = get_swegym_docker_image_name(sample.metadata, subset)
    output_dir = Path("results") / subset / model_name
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"

    env = None
    try:
        # Initialize the Docker environment
        env = DockerEnvironment(
            image=image_name,
            instance_id=instance_id,
            step_timeout=600,
            eval_timeout=600,
        )

        # Off-Load to Thread
        exit_status, messages, eval_report_full, agent_metrics = await asyncio.to_thread(
            run_agent_sync_logic, model, env, problem_statement, sampling_params, sample.metadata, instance_dir, run_id
        )

        # Extract reward from evaluation report
        report_data = eval_report_full.get("eval_report", {}).get(instance_id, {})
        resolved = report_data.get("resolved", False)
        reward = 1.0 if resolved else 0.0

        if len(messages) >= 2:
            sample.prompt = messages[:2]

        state = GenerateState(args)
        tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
            messages=messages,
            tokenizer=state.tokenizer,
        )

        sample.rollout_log_probs = None  # TODO
        sample.tokens = tokens
        sample.loss_mask = loss_mask
        sample.response = response_text
        sample.response_length = response_length
        sample.reward = reward
        sample.metadata["reward"] = reward
        sample.metadata["eval_report"] = eval_report_full
        sample.metadata["messages"] = messages
        sample.metadata["agent_metrics"] = agent_metrics

        if exit_status == "Submitted":
            sample.status = Sample.Status.COMPLETED
        elif exit_status in ("RolloutTruncated", "LimitsExceeded", "CollapseContinued"):
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.ABORTED
            sample.reward = 0.0

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0
    finally:
        if env:
            env.cleanup()

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function - already computed in generate()"""
    reward = sample.metadata.get("reward", 0.0)
    return reward


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter out groups with any aborted samples from training"""
    has_aborted = any(sample.status == Sample.Status.ABORTED for sample in samples)
    if has_aborted:
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging"""
    metrics = {}

    all_metrics = []
    for sample in samples:
        if hasattr(sample, "metadata") and sample.metadata:
            agent_metrics = sample.metadata.get("agent_metrics", {})
            if agent_metrics:
                all_metrics.append(agent_metrics)

    if not all_metrics:
        return {}

    # Count metrics - mean and sum
    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    # Time sum metrics - mean across rollouts
    for key in ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    # Time avg metrics - mean of means
    for key in ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Ratio metrics (all based on total_time which includes eval)
    for key in ["model_time_ratio", "env_time_ratio", "eval_time_ratio"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Total time stats
    values = [m.get("total_time", 0) for m in all_metrics]
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """
    Custom rollout function that wraps sglang_rollout.generate_rollout_async
    and adds agent metrics aggregation.
    """
    from miles.rollout.sglang_rollout import generate_rollout_async as base_generate_rollout_async

    rollout_output, aborted_samples = await base_generate_rollout_async(args, rollout_id, data_source)

    all_samples = []
    for group in rollout_output.samples:
        if isinstance(group[0], list):
            for sample_list in group:
                all_samples.extend(sample_list)
        else:
            all_samples.extend(group)

    agent_metrics = aggregate_agent_metrics(all_samples)

    metrics = rollout_output.metrics or {}
    metrics.update(agent_metrics)

    logger.info(f"Aggregated agent metrics for rollout {rollout_id}: {agent_metrics}")

    return RolloutFnTrainOutput(samples=rollout_output.samples, metrics=metrics), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> tuple[Any, list[list[Sample]]]:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
