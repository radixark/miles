from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.fast.ray.rollout.conftest import make_args, make_sample


def _load_length_reward_module():
    path = Path(__file__).resolve().parents[3] / "examples" / "search-r1" / "length_reward.py"
    spec = importlib.util.spec_from_file_location("search_r1_length_reward", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_penalty_env(monkeypatch):
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_SUCCESS_THRESHOLD", "1.0")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_BETA", "0.10")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_CAP", "0.20")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_REF_QUANTILE", "0.0")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_REL_SLACK", "0.0")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_ABS_SLACK", "0.0")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_SUCCESS_FLOOR", "0.80")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_REQUIRE_COMPLETED", "true")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_LOG_STATS", "false")


def _disable_penalty_env(monkeypatch):
    _set_penalty_env(monkeypatch)
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_BETA", "0.0")
    monkeypatch.setenv("SEARCH_R1_LENGTH_PENALTY_CAP", "0.0")


def test_penalizes_long_successes_within_prompt_group(monkeypatch):
    length_reward = _load_length_reward_module()
    _set_penalty_env(monkeypatch)
    args = make_args(advantage_estimator="ppo", rewards_normalization=False)
    samples = [
        make_sample(group_index=0, index=0, reward=1.0, response_length=10),
        make_sample(group_index=0, index=1, reward=1.0, response_length=30),
        make_sample(group_index=0, index=2, reward=0.0, response_length=100),
    ]

    raw_rewards, shaped_rewards = length_reward.post_process_rewards(args, samples)

    assert raw_rewards == [1.0, 1.0, 0.0]
    assert shaped_rewards == pytest.approx([1.0, 0.8, 0.0])
    assert samples[0].metadata["length_penalty"] == 0.0
    assert samples[1].metadata["length_penalty"] == pytest.approx(0.2)
    assert samples[2].metadata["length_penalty"] == 0.0
    assert samples[2].metadata["length_success"] == 0.0


def test_skips_groups_without_two_successes(monkeypatch):
    length_reward = _load_length_reward_module()
    _set_penalty_env(monkeypatch)
    args = make_args(advantage_estimator="ppo", rewards_normalization=False)
    samples = [
        make_sample(group_index=0, index=0, reward=1.0, response_length=10),
        make_sample(group_index=0, index=1, reward=0.0, response_length=100),
    ]

    _raw_rewards, shaped_rewards = length_reward.post_process_rewards(args, samples)

    assert shaped_rewards == pytest.approx([1.0, 0.0])
    assert samples[0].metadata["length_group_eligible"] == 0.0
    assert samples[1].metadata["length_group_eligible"] == 0.0


def test_uses_loss_mask_as_effective_length(monkeypatch):
    length_reward = _load_length_reward_module()
    _set_penalty_env(monkeypatch)
    args = make_args(advantage_estimator="ppo", rewards_normalization=False)
    samples = [
        make_sample(group_index=0, index=0, reward=1.0, response_length=10),
        make_sample(group_index=0, index=1, reward=1.0, response_length=100),
    ]
    samples[0].loss_mask = [1] * 10
    samples[1].loss_mask = [1] * 20 + [0] * 80

    _raw_rewards, shaped_rewards = length_reward.post_process_rewards(args, samples)

    assert shaped_rewards == pytest.approx([1.0, 0.9])
    assert samples[1].metadata["length_effective_response_length"] == 20


def test_normalizes_by_actual_prompt_group_for_interleaved_samples(monkeypatch):
    length_reward = _load_length_reward_module()
    _disable_penalty_env(monkeypatch)
    args = make_args(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
    )
    samples = [
        make_sample(group_index=0, index=0, reward=1.0),
        make_sample(group_index=1, index=1, reward=10.0),
        make_sample(group_index=0, index=2, reward=3.0),
        make_sample(group_index=1, index=3, reward=30.0),
    ]

    _raw_rewards, shaped_rewards = length_reward.post_process_rewards(args, samples)

    assert shaped_rewards == pytest.approx([-1.0, -10.0, 1.0, 10.0])


def test_std_normalization_single_sample_group_does_not_produce_nan(monkeypatch):
    length_reward = _load_length_reward_module()
    _disable_penalty_env(monkeypatch)
    args = make_args(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
    )
    samples = [make_sample(group_index=0, index=0, reward=1.0)]

    _raw_rewards, shaped_rewards = length_reward.post_process_rewards(args, samples)

    assert shaped_rewards == pytest.approx([0.0])
