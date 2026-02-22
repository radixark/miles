import argparse
import asyncio
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.swegym_runner import get_swegym_docker_image_name, run_eval

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _status_from_exit_status(exit_status: str) -> Sample.Status:
    if exit_status == "Submitted":
        return Sample.Status.COMPLETED
    if exit_status in ("RolloutTruncated", "LimitsExceeded", "CollapseContinued"):
        return Sample.Status.TRUNCATED
    return Sample.Status.ABORTED


def _decorate_sample(
    sample: Sample,
    *,
    status: Sample.Status,
    reward: float,
    eval_report_full: dict[str, Any],
    messages: list[dict[str, Any]],
    agent_metrics: dict[str, Any],
) -> Sample:
    sample.status = status
    sample.reward = 0.0 if status == Sample.Status.ABORTED else reward

    metadata = dict(sample.metadata or {})
    metadata["reward"] = sample.reward
    metadata["eval_report"] = eval_report_full
    metadata["messages"] = messages
    metadata["agent_metrics"] = agent_metrics
    sample.metadata = metadata

    return sample


def _normalize_eval_instance_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    instance_id = normalized.get("instance_id")
    if isinstance(instance_id, str):
        normalized["instance_id"] = instance_id.lower()

    repo = normalized.get("repo")
    if isinstance(repo, str):
        normalized["repo"] = repo.lower()
    elif isinstance(instance_id, str):
        # SWE-Gym instance ids are usually "<org>__<repo>-<id>"; derive a repo hint for parsers.
        repo_from_instance = instance_id.rsplit("-", 1)[0].replace("__", "/").lower()
        normalized["repo"] = repo_from_instance

    return normalized


def _get_eval_entry(eval_report_full: dict[str, Any], instance_id: str) -> dict[str, Any]:
    report = eval_report_full.get("eval_report", {})
    if not isinstance(report, dict):
        return {}

    if instance_id in report:
        return report[instance_id]

    lower_instance_id = instance_id.lower()
    if lower_instance_id in report:
        return report[lower_instance_id]

    return {}


def _build_agent_config(step_limit: int, collapse_limit: int) -> dict[str, Any]:
    # Keep agent prompting minimal/explicit to avoid over-constraining generations.
    return {
        "step_limit": step_limit,
        "collapse_limit": collapse_limit,
    }


def _build_environment_config() -> dict[str, Any]:
    # Keep SWE command execution rooted at /testbed and deterministic shell behavior.
    return {
        "cwd": "/testbed",
        "env": {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
    }


def run_agent_sync_logic(
    model,
    env,
    problem_statement,
    sampling_params,
    metadata,
    instance_dir,
    run_id,
    agent_config: dict[str, Any],
):
    """Blocking SWE-agent run + evaluation; executed in a thread."""
    agent = DefaultAgent(
        model=model,
        env=env,
        responses_create_params={"input": []},
        sampling_params=sampling_params,
        **agent_config,
    )

    exit_status, result_patch, agent_metrics = agent.run(problem_statement)

    eval_start = time.time()
    eval_instance = _normalize_eval_instance_metadata(metadata)
    eval_instance_id = str(eval_instance.get("instance_id", metadata.get("instance_id", "unknown")))
    try:
        eval_report_full = run_eval(
            instance=eval_instance,
            env=env,
            model_patch=result_patch,
            instance_dir=instance_dir,
            run_id=run_id,
        )
    except KeyError as exc:
        parser_key = str(exc).strip("'\"")
        retry_instance = dict(eval_instance)
        if "/" in parser_key:
            retry_instance["repo"] = parser_key.lower()
        try:
            eval_report_full = run_eval(
                instance=retry_instance,
                env=env,
                model_patch=result_patch,
                instance_dir=instance_dir,
                run_id=run_id,
            )
            logger.info(
                "SWE eval parser recovered after normalization for instance %s (parser_key=%s)",
                eval_instance_id,
                parser_key,
            )
        except KeyError:
            instance_id = str(metadata.get("instance_id", "unknown"))
            logger.info(
                "SWE eval parser unavailable for instance %s: %s (normalized_instance_id=%s, normalized_repo=%s)",
                instance_id,
                exc,
                eval_instance.get("instance_id"),
                eval_instance.get("repo"),
            )
            eval_report_full = {
                "eval_report": {
                    instance_id: {
                        "resolved": False,
                        "error": f"eval_parser_unavailable: {exc}",
                    }
                }
            }
    eval_time = time.time() - eval_start

    agent_metrics["eval_time"] = eval_time
    total_time = agent_metrics.get("agent_run_time", 0.0) + eval_time
    agent_metrics["total_time"] = total_time
    agent_metrics["model_time_ratio"] = agent_metrics.get("model_query_time_sum", 0.0) / max(total_time, 1e-6)
    agent_metrics["env_time_ratio"] = agent_metrics.get("env_execution_time_sum", 0.0) / max(total_time, 1e-6)
    agent_metrics["eval_time_ratio"] = eval_time / max(total_time, 1e-6)

    return exit_status, agent.messages, eval_report_full, agent_metrics


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """
    Run one SWE task end-to-end with mini-swe-agent and return a rollout sample.

    Flow:
    1) Create an OAI session on Miles router.
    2) Run agent + SWE eval in a worker thread.
    3) Reconstruct training tokens/logprobs from OAI session records.
    4) Attach reward/status/metrics metadata.
    """
    args = input.args
    source_sample = input.sample

    assert not args.partial_rollout, "Partial rollout is not supported for SWE-agent generation."

    metadata = dict(source_sample.metadata or {})
    instance_id = str(metadata.get("instance_id", f"sample-{source_sample.index}"))
    subset = metadata.get("subset", "gym")
    problem_statement = metadata.get("problem_statement")
    if not problem_statement and isinstance(source_sample.prompt, str):
        problem_statement = source_sample.prompt

    if not problem_statement:
        logger.error("Missing problem statement for instance %s", instance_id)
        failed = deepcopy(source_sample)
        failed.status = Sample.Status.ABORTED
        failed.reward = 0.0
        return GenerateFnOutput(samples=failed)

    model_name = f"sglang/{Path(args.hf_checkpoint).name}"
    output_dir = Path("results") / subset / model_name
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
    agent_config = _build_agent_config(args.generate_step_limit, args.generate_collapse_limit)
    env_config = _build_environment_config()

    tracer: OpenAIEndpointTracer | None = None
    env = None
    messages: list[dict[str, Any]] = []
    eval_report_full: dict[str, Any] = {}
    agent_metrics: dict[str, Any] = {}

    try:
        tracer = await OpenAIEndpointTracer.create(args)

        model_config = {
            "model_name": model_name,
            "model_kwargs": {
                "base_url": f"{tracer.base_url}/v1",
                "api_key": "dummy",
            },
        }
        model = get_model(model_name, config=model_config)

        image_name = get_swegym_docker_image_name(metadata, subset)
        env = DockerEnvironment(
            image=image_name,
            instance_id=instance_id,
            step_timeout=600,
            eval_timeout=600,
            **env_config,
        )

        request_kwargs = build_chat_request_kwargs(input.sampling_params)
        # Keep stop behavior explicit for SWE multi-turn tracing: stop at role boundary and
        # do not preserve raw stop tokens in assistant content.
        request_kwargs["no_stop_trim"] = False
        exit_status, messages, eval_report_full, agent_metrics = await asyncio.to_thread(
            run_agent_sync_logic,
            model,
            env,
            problem_statement,
            request_kwargs,
            metadata,
            instance_dir,
            run_id,
            agent_config,
        )

        status = _status_from_exit_status(exit_status)

        eval_data = _get_eval_entry(eval_report_full, instance_id)
        reward = 1.0 if eval_data.get("resolved", False) else 0.0

        records = await tracer.collect_records()
        tracer = None  # collect_records already attempts session deletion.
        if not records:
            raise RuntimeError(f"No OAI session records collected for instance {instance_id}")

        traced_samples = compute_samples_from_openai_records(source_sample, records, input.state.tokenizer)
        merged_sample = merge_samples(traced_samples, input.state.tokenizer)
        output_sample = _decorate_sample(
            merged_sample,
            status=status,
            reward=reward,
            eval_report_full=eval_report_full,
            messages=messages,
            agent_metrics=agent_metrics,
        )

        return GenerateFnOutput(samples=output_sample)

    except Exception as exc:
        logger.error("Error processing instance %s: %s", instance_id, exc, exc_info=True)
        failed = deepcopy(source_sample)
        failed = _decorate_sample(
            failed,
            status=Sample.Status.ABORTED,
            reward=0.0,
            eval_report_full=eval_report_full,
            messages=messages,
            agent_metrics=agent_metrics,
        )
        return GenerateFnOutput(samples=failed)
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                logger.warning("Failed to cleanup DockerEnvironment for instance %s", instance_id)
        if tracer is not None:
            try:
                await post(f"{tracer.router_url}/sessions/{tracer.session_id}", {}, action="delete")
            except Exception:
                logger.warning("Failed to cleanup OAI session for instance %s", instance_id)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function compatibility hook; reward is populated during generation."""
    if sample.reward is not None and isinstance(sample.reward, (float, int)):
        return float(sample.reward)
    return float(sample.metadata.get("reward", 0.0))


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter out groups with any aborted samples from training."""
    has_aborted = any(sample.status == Sample.Status.ABORTED for sample in samples)
    if has_aborted:
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")

    return DynamicFilterOutput(keep=True)


def aggregate_agent_metrics(samples: list[Sample]) -> dict[str, float]:
    """Aggregate per-sample agent metrics for rollout logging."""
    all_metrics: list[dict[str, Any]] = []
    for sample in samples:
        if sample.metadata and sample.metadata.get("agent_metrics"):
            all_metrics.append(sample.metadata["agent_metrics"])

    if not all_metrics:
        return {}

    metrics: dict[str, float] = {}

    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    for key in ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"]:
        values = [m.get(key, 0.0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    for key in ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"]:
        values = [m.get(key, 0.0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    for key in ["model_time_ratio", "env_time_ratio", "eval_time_ratio"]:
        values = [m.get(key, 0.0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    total_time_values = [m.get("total_time", 0.0) for m in all_metrics]
    if total_time_values:
        metrics["agent/total_time_mean"] = sum(total_time_values) / len(total_time_values)
        metrics["agent/total_time_max"] = max(total_time_values)
        metrics["agent/total_time_min"] = min(total_time_values)

    return metrics


def aggregate_reward_diagnostics(samples: list[Sample]) -> dict[str, float]:
    """Aggregate reward outcome breakdown for faster rollout debugging."""
    resolved_count = 0
    unresolved_count = 0
    parser_unavailable_count = 0

    for sample in samples:
        metadata = sample.metadata or {}
        instance_id = str(metadata.get("instance_id", ""))
        eval_report_full = metadata.get("eval_report", {})
        eval_data = _get_eval_entry(eval_report_full, instance_id) if instance_id else {}

        if eval_data.get("resolved", False):
            resolved_count += 1
            continue

        unresolved_count += 1
        error = eval_data.get("error")
        if isinstance(error, str) and "eval_parser_unavailable" in error:
            parser_unavailable_count += 1

    total = len(samples)
    metrics: dict[str, float] = {
        "reward/resolved_count": float(resolved_count),
        "reward/unresolved_count": float(unresolved_count),
        "reward/parser_unavailable_count": float(parser_unavailable_count),
    }
    if total > 0:
        metrics["reward/resolved_ratio"] = resolved_count / total
        metrics["reward/unresolved_ratio"] = unresolved_count / total
        metrics["reward/parser_unavailable_ratio"] = parser_unavailable_count / total

    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Inject SWE agent metrics into default rollout logging."""
    metrics = aggregate_agent_metrics(samples)
    metrics.update(aggregate_reward_diagnostics(samples))
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(metrics)
    return False


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-step-limit", type=int, default=250)
    parser.add_argument("--generate-collapse-limit", type=int, default=3)


generate.add_arguments = _add_arguments
