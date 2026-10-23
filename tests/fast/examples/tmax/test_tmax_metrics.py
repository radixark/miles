"""TMax's per-rollout metrics (tmax_metrics): groups recorded by the filter hooks, then one log call per batch."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import tmax_metrics  # noqa: E402

from miles.utils.types import Sample  # noqa: E402

ARGS = SimpleNamespace(
    reward_key=None,
    rollout_batch_size=2,
    n_samples_per_prompt=4,
    wandb_always_use_train_step=False,
    custom_reward_post_process_path=None,
)


def _episode(end_reason="submitted", turns=10, tool_calls=10, tool_errors=1, start=2.0):
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "end_reason": end_reason,
        "submitted": float(end_reason == "submitted"),
        "start_failed": float(end_reason == "start_failed"),
        "model_query_time_sum": 30.0,
        "env_execution_time_sum": 20.0,
        "total_time": 60.0,
        "sandbox_start_time": start,
    }


def _group(index, rewards, lengths=None, **episode_kwargs):
    lengths = lengths or [100] * len(rewards)
    return [
        Sample(group_index=index, reward=r, response_length=n, metadata={"agent_metrics": _episode(**episode_kwargs)})
        for r, n in zip(rewards, lengths, strict=True)
    ]


@pytest.fixture
def logged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tmax_metrics.tracking, "log", lambda args, metrics, step_key: calls.append((metrics, step_key))
    )
    tmax_metrics._drain()  # start empty
    return calls


def test_one_log_call_per_batch_with_every_family(logged):
    mixed = _group(0, [1.0, 0.0, 1.0, 0.0], lengths=[300, 100, 500, 100])
    all_fail = _group(1, [0.0, 0.0, 0.0, 0.0], end_reason="max_steps", turns=64, tool_calls=64)
    all_pass = _group(2, [1.0, 1.0, 1.0, 1.0])
    for group in (mixed, all_fail, all_pass):
        tmax_metrics.nonzero_std_filter(ARGS, group)

    assert tmax_metrics.log_rollout_data(3, ARGS, mixed, {}, 1.0) is False, "Miles's own metrics still get logged"
    ((metrics, step_key),) = logged
    assert step_key == "rollout/step" and metrics["rollout/step"] == 3
    assert all(k.startswith("rollout/") for k in metrics), "W&B plots rollout/* against rollout/step"

    # Before the filter: three groups, the all-fail and all-pass ones included.
    assert metrics["rollout/unfiltered/groups"] == 3
    assert metrics["rollout/unfiltered/frac_solve_rate_0"] == pytest.approx(1 / 3)
    assert metrics["rollout/unfiltered/frac_solve_rate_1"] == pytest.approx(1 / 3)
    assert metrics["rollout/unfiltered/frac_solve_rate_25_50"] == pytest.approx(1 / 3)
    # Agent metrics over all 12 episodes, filtered groups included.
    assert metrics["rollout/agent/episodes"] == 12
    assert metrics["rollout/agent/end_reason/max_steps"] == pytest.approx(4 / 12)
    assert metrics["rollout/agent/non_submitting_fraction"] == pytest.approx(4 / 12)
    assert metrics["rollout/tools/bash/avg_calls_per_rollout"] == pytest.approx((8 * 10 + 4 * 64) / 12)
    assert metrics["rollout/tools/env_reset/avg_runtime"] == pytest.approx(2.0)
    # Trained samples: the mixed group only.
    assert metrics["rollout/trained/response_len_solved_mean"] == 400
    assert metrics["rollout/trained/response_len_unsolved_mean"] == 100
    assert metrics["rollout/trained/advantage_min"] == pytest.approx(-0.5)
    assert metrics["rollout/trained/advantage_max"] == pytest.approx(0.5)
    assert metrics["rollout/trained/samples_total"] == 4 * 2 * 4


def test_the_filter_wrapper_keeps_miles_decision():
    assert tmax_metrics.nonzero_std_filter(ARGS, _group(0, [1.0, 0.0])).keep
    assert not tmax_metrics.nonzero_std_filter(ARGS, _group(1, [1.0, 1.0])).keep
    assert tmax_metrics.missing_reward_filter(ARGS, _group(2, [1.0, 1.0])).keep
    assert not tmax_metrics.missing_reward_filter(
        ARGS, _group(3, [1.0, None])
    ).keep, "a missing reward drops the group"


def test_recorded_groups_are_logged_once(logged):
    tmax_metrics.missing_reward_filter(ARGS, _group(0, [1.0, 0.0]))
    tmax_metrics.log_rollout_data(0, ARGS, [], {}, 1.0)
    tmax_metrics.log_rollout_data(1, ARGS, [], {}, 1.0)
    assert "rollout/unfiltered/groups" in logged[0][0]
    assert len(logged) == 1, "nothing new was recorded for the second batch"


def test_session_server_v2_groups_count_every_episode(logged):
    """With session server v2 a group holds a list[Sample] per episode, not a Sample (2026-10-20, the MoE smoke)."""
    group = [[s] for s in _group(0, [1.0, 0.0, 1.0, 0.0], end_reason="max_steps", turns=8)]
    tmax_metrics.missing_reward_filter(ARGS, group)
    tmax_metrics.log_rollout_data(0, ARGS, [], {}, 1.0)
    metrics = logged[0][0]
    assert metrics["rollout/agent/episodes"] == 4
    assert metrics["rollout/agent/turns_mean"] == 8
    assert metrics["rollout/unfiltered/solve_rate_mean"] == pytest.approx(0.5)


def test_without_the_filter_hooks_agent_metrics_come_from_the_trained_samples(logged):
    samples = _group(0, [1.0, 0.0], end_reason="start_failed")
    tmax_metrics.log_rollout_data(0, ARGS, samples, {}, 1.0)
    metrics = logged[0][0]
    assert metrics["rollout/agent/episodes"] == 2
    assert metrics["rollout/tools/env_reset/failure_rate"] == 1.0
    assert "rollout/unfiltered/groups" not in metrics


def test_missing_rewards_are_left_out():
    samples = _group(0, [None, 1.0, 0.0, None])
    metrics = tmax_metrics._trained_metrics(ARGS, 0, samples)
    assert metrics["rollout/trained/advantage_max"] == pytest.approx(0.5)


def test_pure_opd_logs_no_reward_advantage():
    """Pure OPD trains on reward 0 (generate.opd_pure_rewards): the centered task reward is not the advantage."""
    args = SimpleNamespace(**{**vars(ARGS), "custom_reward_post_process_path": "generate.opd_pure_rewards"})
    metrics = tmax_metrics._trained_metrics(args, 0, _group(0, [1.0, 0.0]))
    assert metrics["rollout/trained/response_len_solved_mean"] == 100
    assert not any(k.startswith("rollout/trained/advantage") for k in metrics)
