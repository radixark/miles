from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.rollout.group_relative_efficiency import (
    shape_trajectory_means,
    trajectory_token_cost,
)
from miles.rollout.polar_reward import post_process_rewards


def _sample(*, output: int, input_tokens: int = 0, response_length: int = 1):
    return SimpleNamespace(
        response_length=response_length,
        metadata={
            "polar": {
                "trajectory_metadata": {
                    "evaluation": {
                        "raw_flow": {
                            "token_usage": {"input": input_tokens, "output": output}
                        }
                    }
                }
            }
        },
    )


def test_token_cost_uses_flow_usage_once_for_multi_trace_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLAR_GROUP_RELATIVE_INPUT_WEIGHT", "0.1")
    samples = [_sample(output=100, input_tokens=200), _sample(output=100, input_tokens=200)]
    assert trajectory_token_cost(samples, [0, 1]) == 120.0


def test_relative_efficiency_only_rewards_no_worse_candidate() -> None:
    shaped = shape_trajectory_means(
        {"short-wrong": 0.2, "long-correct": 0.8, "short-correct": 0.8},
        {"short-wrong": 10.0, "long-correct": 100.0, "short-correct": 20.0},
        weight=0.1,
    )
    assert shaped["short-wrong"] == 0.2
    assert shaped["long-correct"] == 0.8
    assert shaped["short-correct"] == pytest.approx(0.81)


def test_postprocessor_records_efficiency_audit_metadata() -> None:
    samples = [
        _sample(output=100, response_length=1),
        _sample(output=200, response_length=1),
    ]
    for index, sample in enumerate(samples):
        sample.group_index = 7
        sample.rollout_id = index
        sample.reward = {"score": 0.8}
        sample.loss_mask = [1]
        sample.metadata["polar"]["trajectory_metadata"]["evaluation"]["raw_flow"][
            "token_usage"
        ] = {"input": 0, "output": 100 if index == 0 else 200}

    args = type(
        "Args",
        (),
        {
            "polar_reward_key": "score",
            "rewards_normalization": True,
            "advantage_estimator": "grpo",
            "grpo_std_normalization": False,
            "polar_group_relative_token_weight": 0.1,
        },
    )()
    post_process_rewards(args, samples)
    audit = samples[0].metadata["polar"]["group_relative_efficiency"]
    assert audit["enabled"] is True
    assert audit["weight"] == 0.1
    assert audit["bonus"] == pytest.approx(0.02)
