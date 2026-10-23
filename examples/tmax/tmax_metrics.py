"""TMax's per-rollout W&B metrics beyond Miles's own, the ones open-instruct's runs log.

Two hooks, both running in the rollout process:

* `nonzero_std_filter` / `missing_reward_filter` (``--dynamic-sampling-filter-path``) see every finished group before
  Miles's zero-variance filter decides, so they record the group's solve rate and its episodes' agent metrics;
* `log_rollout_data` (``--custom-rollout-log-function-path``) runs once per trained batch, in synchronous and in
  fully-async mode alike, and logs what was recorded since the previous batch together with metrics of the trained
  samples. It returns False, so Miles's own rollout metrics are logged as before.

Every key is under ``rollout/``, which W&B plots against ``rollout/step``:

* ``rollout/agent/*``: end reasons, turns, tool calls, model and environment time, over every finished episode
  (filtered groups included); ``non_submitting_fraction`` is open-instruct's ``val/non_submitting_completion_fraction``;
* ``rollout/tools/{bash,env_reset}/*``: open-instruct's ``tools/*`` (calls per episode, runtime, failure rate);
* ``rollout/unfiltered/*``: the groups' solve rates before the filter, open-instruct's ``batch/percent_solved_*``;
* ``rollout/trained/*``: response lengths of solved and unsolved trained samples (``val/sequence_lengths_*``),
  advantages (``val/advantages_*``) and samples trained so far (``episode``).
"""

import threading
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean

from miles.rollout.filter_hub.base_types import FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter, apply_reward_nonzero_std_filter
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

# Groups finished since the last logged batch: (solve rate, agent metrics of each episode). The filter and the log
# hook run in the same rollout process; the lock guards against them running on different threads.
_lock = threading.Lock()
_solve_rates: list[float] = []
_episodes: list[dict] = []

_SOLVE_RATE_BINS = (("0_25", 0.0, 0.25), ("25_50", 0.25, 0.5), ("50_75", 0.5, 0.75), ("75_100", 0.75, 1.0))
_MEAN_KEYS = (
    "turns",
    "tool_calls",
    "model_query_time_sum",
    "env_execution_time_sum",
    "total_time",
    "sandbox_start_time",
)


def nonzero_std_filter(args, samples, **kwargs) -> FilterOutput:
    """Miles's zero-variance filter, after recording the group (open-instruct's --active_sampling)."""
    _record_group(args, samples)
    return apply_reward_nonzero_std_filter(args, samples, **kwargs)


def missing_reward_filter(args, samples, **kwargs) -> FilterOutput:
    """Miles's missing-reward filter alone, after recording the group (filter_zero_std off, as in smoke tests): a
    group with a sample the sandbox failed to score (reward None) is dropped rather than trained on a false 0."""
    _record_group(args, samples)
    return apply_missing_reward_filter(args, samples, **kwargs)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    solve_rates, episodes = _drain()
    if not episodes:  # the filter hooks are not installed: describe the trained episodes instead
        episodes = [s.metadata["agent_metrics"] for s in samples if (s.metadata or {}).get("agent_metrics")]
    metrics = _agent_metrics(episodes) | _unfiltered_metrics(solve_rates) | _trained_metrics(args, rollout_id, samples)
    if metrics:
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key="rollout/step")
    return False


def _record_group(args, samples) -> None:
    rewards = [_reward(args, s) for s in iter_samples(samples)]
    rewards = [r for r in rewards if r is not None]
    episodes = [s.metadata["agent_metrics"] for s in iter_samples(samples) if (s.metadata or {}).get("agent_metrics")]
    with _lock:
        if rewards:
            _solve_rates.append(sum(r > 0 for r in rewards) / len(rewards))
        _episodes.extend(episodes)


def _drain() -> tuple[list[float], list[dict]]:
    with _lock:
        solve_rates, episodes = list(_solve_rates), list(_episodes)
        _solve_rates.clear()
        _episodes.clear()
    return solve_rates, episodes


def _reward(args, sample: Sample) -> float | None:
    if sample.reward is None:
        return None
    value = sample.get_reward_value(args)
    return None if value is None else float(value)


def _agent_metrics(episodes: list[dict]) -> dict[str, float]:
    if not episodes:
        return {}
    metrics = {"rollout/agent/episodes": len(episodes)}
    for key in _MEAN_KEYS:
        values = [e[key] for e in episodes if key in e]
        if values:
            metrics[f"rollout/agent/{key}_mean"] = mean(values)
    for reason, count in Counter(e.get("end_reason") for e in episodes).items():
        metrics[f"rollout/agent/end_reason/{reason}"] = count / len(episodes)
    metrics["rollout/agent/non_submitting_fraction"] = 1.0 - mean(e.get("submitted", 0.0) for e in episodes)
    # Gateway backend: why the fleet removed a container (memory_limit, pids_limit, affinity_owner_lost, ...).
    for reason, count in Counter(e["sandbox_lost_reason"] for e in episodes if "sandbox_lost_reason" in e).items():
        metrics[f"rollout/agent/sandbox_lost_reason/{reason}"] = count / len(episodes)

    calls = sum(e.get("tool_calls", 0) for e in episodes)
    metrics["rollout/tools/bash/avg_calls_per_rollout"] = calls / len(episodes)
    if calls:
        metrics["rollout/tools/bash/avg_runtime"] = sum(e.get("env_execution_time_sum", 0.0) for e in episodes) / calls
        metrics["rollout/tools/bash/failure_rate"] = sum(e.get("tool_errors", 0) for e in episodes) / calls
    starts = [e for e in episodes if "sandbox_start_time" in e]
    if starts:
        metrics["rollout/tools/env_reset/avg_runtime"] = mean(e["sandbox_start_time"] for e in starts)
        metrics["rollout/tools/env_reset/failure_rate"] = mean(e.get("start_failed", 0.0) for e in starts)
    return metrics


def _unfiltered_metrics(solve_rates: list[float]) -> dict[str, float]:
    if not solve_rates:
        return {}
    n = len(solve_rates)
    metrics = {
        "rollout/unfiltered/groups": n,
        "rollout/unfiltered/solve_rate_mean": mean(solve_rates),
        "rollout/unfiltered/frac_solve_rate_0": sum(r == 0 for r in solve_rates) / n,
        "rollout/unfiltered/frac_solve_rate_1": sum(r == 1 for r in solve_rates) / n,
    }
    for name, low, high in _SOLVE_RATE_BINS:  # open bins: all-fail and all-pass groups are counted above
        metrics[f"rollout/unfiltered/frac_solve_rate_{name}"] = sum(low < r <= high and r < 1 for r in solve_rates) / n
    return metrics


def _trained_metrics(args, rollout_id: int, samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {}
    metrics = {"rollout/trained/samples_total": (rollout_id + 1) * args.rollout_batch_size * args.n_samples_per_prompt}
    scored = [(s, r) for s in samples if (r := _reward(args, s)) is not None]
    for name, chosen in (("solved", [s for s, r in scored if r > 0]), ("unsolved", [s for s, r in scored if r <= 0])):
        if chosen:
            lengths = [s.effective_response_length for s in chosen]
            metrics[f"rollout/trained/response_len_{name}_mean"] = mean(lengths)
            metrics[f"rollout/trained/response_len_{name}_max"] = max(lengths)
    # A reward post-process (pure OPD's) replaces the trained reward, so the centered task reward is not the advantage.
    advantages = list(_centered_advantages(scored)) if args.custom_reward_post_process_path is None else []
    if advantages:
        metrics["rollout/trained/advantage_min"] = min(advantages)
        metrics["rollout/trained/advantage_max"] = max(advantages)
        metrics["rollout/trained/advantage_abs_mean"] = mean(abs(a) for a in advantages)
    return metrics


def _centered_advantages(scored: list[tuple[Sample, float]]) -> Iterable[float]:
    """Reward minus its group's mean: the advantage GRPO without std normalization trains on (as open-instruct's
    advantage_normalization_type centered)."""
    groups = defaultdict(list)
    for sample, reward in scored:
        groups[sample.group_index].append(reward)
    for rewards in groups.values():
        group_mean = mean(rewards)
        yield from (r - group_mean for r in rewards)
