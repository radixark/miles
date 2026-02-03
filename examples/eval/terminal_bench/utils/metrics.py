from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger("terminal_bench_server")


def extract_tb_metrics(metrics_path: Path) -> dict[str, Any]:
    try:
        with open(metrics_path, encoding="utf-8") as fp:
            metrics_data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", metrics_path, exc)
        return {}

    metrics: dict[str, Any] = {}

    # core metrics
    accuracy = metrics_data.get("accuracy")
    if isinstance(accuracy, (int, float)):
        metrics["accuracy"] = float(accuracy)

    n_resolved = metrics_data.get("n_resolved")
    if isinstance(n_resolved, (int, float)):
        metrics["n_resolved"] = int(n_resolved)

    n_unresolved = metrics_data.get("n_unresolved")
    if isinstance(n_unresolved, (int, float)):
        metrics["n_unresolved"] = int(n_unresolved)

    # pass@k flatten
    pass_at_k = metrics_data.get("pass_at_k")
    if isinstance(pass_at_k, dict):
        for k, v in pass_at_k.items():
            if isinstance(v, (int, float)):
                metrics[f"pass_at_k/{k}"] = float(v)

    # token stats from per-task results
    results = metrics_data.get("results")
    if isinstance(results, list):
        input_tokens = [
            r.get("total_input_tokens")
            for r in results
            if isinstance(r, dict)
            and isinstance(r.get("total_input_tokens"), (int, float))
        ]
        output_tokens = [
            r.get("total_output_tokens")
            for r in results
            if isinstance(r, dict)
            and isinstance(r.get("total_output_tokens"), (int, float))
        ]

        if input_tokens:
            metrics["total_input_tokens_mean"] = float(statistics.mean(input_tokens))
            metrics["total_input_tokens_median"] = float(
                statistics.median(input_tokens)
            )
            metrics["total_input_tokens_min"] = float(min(input_tokens))
            metrics["total_input_tokens_max"] = float(max(input_tokens))
        if output_tokens:
            metrics["total_output_tokens_mean"] = float(
                statistics.mean(output_tokens)
            )
            metrics["total_output_tokens_median"] = float(
                statistics.median(output_tokens)
            )

    return metrics


def extract_harbor_metrics(
    metrics_path: Path,
    run_dir: Path,
    *,
    model_name: str,
    dataset_name: str,
    agent_name: str,
) -> dict[str, Any]:
    try:
        with open(metrics_path, encoding="utf-8") as fp:
            metrics_data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", metrics_path, exc)
        return {}

    evals = metrics_data.get("stats", {}).get("evals", {})
    if not isinstance(evals, dict) or not evals:
        return {}

    candidates = (
        f"{agent_name}__{model_name}__{dataset_name}",
        f"{agent_name}__{model_name}__terminal-bench",
    )
    entry = next((evals.get(key) for key in candidates if key in evals), None)
    if entry is None:
        entry = next(iter(evals.values()))
    if not isinstance(entry, dict):
        return {}

    metrics: dict[str, Any] = {}
    for key in ("n_trials", "n_errors"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = int(value)

    metrics_block = entry.get("metrics")
    if isinstance(metrics_block, list):
        for metric in metrics_block:
            if isinstance(metric, dict):
                for name, value in metric.items():
                    if isinstance(value, (int, float)):
                        metrics[name] = float(value)

    reward_stats = entry.get("reward_stats")
    if isinstance(reward_stats, dict):
        for reward_name, reward_values in reward_stats.items():
            if isinstance(reward_values, dict):
                for reward_value, trials in reward_values.items():
                    if isinstance(trials, list):
                        metrics[f"reward_stats/{reward_name}/{reward_value}"] = len(
                            trials
                        )

    exception_stats = entry.get("exception_stats")
    if isinstance(exception_stats, dict):
        for exception_name, trials in exception_stats.items():
            if isinstance(trials, list):
                metrics[f"exception_stats/{exception_name}"] = len(trials)

    input_tokens = []
    output_tokens = []
    for result_path in run_dir.glob("*/result.json"):
        try:
            with open(result_path, encoding="utf-8") as fp:
                task_data = json.load(fp)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s", result_path)
            continue
        agent_result = task_data.get("agent_result") or {}
        n_input = agent_result.get("n_input_tokens")
        if isinstance(n_input, (int, float)):
            input_tokens.append(float(n_input))
        n_output = agent_result.get("n_output_tokens")
        if isinstance(n_output, (int, float)):
            output_tokens.append(float(n_output))

    def add_token_stats(name: str, values: list[float]) -> None:
        if not values:
            return
        metrics[f"{name}/min"] = float(min(values))
        metrics[f"{name}/max"] = float(max(values))
        metrics[f"{name}/mean"] = float(statistics.mean(values))
        metrics[f"{name}/median"] = float(statistics.median(values))

    add_token_stats("n_input_tokens", input_tokens)
    add_token_stats("n_output_tokens", output_tokens)

    return metrics
