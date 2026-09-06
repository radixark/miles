"""Exercise chess hooks through the default Miles postprocessor and metric logger."""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

import chess_training
from chess_filter import check_chess_group
from miles.ray.rollout.metrics import log_rollout_data
from miles.ray.rollout.repetition import apply_repetition_reward_penalty
from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.utils.types import Sample


def _postprocess_game(
    *, invalid: bool, reward: float, texts: list[str], rollout_id: int = 0
) -> list[Sample]:
    samples = [
        make_sample(
            index=rollout_id,
            rollout_id=rollout_id,
            response=text,
            response_length=2,
            loss_mask=[1, 1],
            metadata={"leaf": {"node_id": i, "path_node_ids": [i]}},
        )
        for i, text in enumerate(texts)
    ]
    metadata = {
        "tree": {"nodes": [{"id": i, "completion_span": [0, 2]} for i in range(len(samples))]},
        "agent": {
            "reward": reward,
            "repetition_reward_penalty": 0.5,
            "chess_result": {
                "invalid_move_termination": invalid,
                "outcome": "loss" if invalid else "win",
            },
        },
    }
    old_samples = deepcopy(samples)
    result = chess_training.postprocess_samples(samples, metadata)
    for old, new in zip(old_samples, result, strict=True):
        assert new.tokens == old.tokens
        assert new.loss_mask == old.loss_mask
    return result


@pytest.mark.parametrize("text", ["", "thinking without an answer", "!" * 15000])
def test_invalid_move_reward_remains_exactly_zero_in_training(text: str) -> None:
    failed = _postprocess_game(invalid=True, reward=0.0, texts=[text])
    successful = _postprocess_game(invalid=False, reward=1.0, texts=["<move>e2e4</move>"], rollout_id=1)
    samples = failed + successful
    args = make_args(n_samples_per_prompt=2, rollout_batch_size=1, repetition_reward_penalty=0)
    apply_repetition_reward_penalty(args, samples)
    raw, normalized = _post_process_rewards(args, samples, None)
    assert raw == [0.0, 1.0]
    assert normalized[0] < 0 < normalized[1]
    assert samples[0].metadata["raw_reward"] == 0.0
    assert samples[0].metadata["repetition_reward_penalty_applied"] == 0.0
    assert check_chess_group(args, samples).keep is True


@pytest.mark.parametrize("base_reward", [0.0, 1.0])
def test_non_error_games_keep_repetition_penalty_across_compaction(base_reward: float) -> None:
    samples = _postprocess_game(
        invalid=False, reward=base_reward, texts=["!" * 15000, "<move>e2e4</move>"]
    )
    assert [s.reward for s in samples] == [base_reward - 0.5] * 2
    assert [s.metadata["has_repetition"] for s in samples] == [True, False]
    assert [s.metadata["repetition_reward_penalty_applied"] for s in samples] == [0.5, 0.5]


def test_failure_metric_reaches_standard_tracking_once_per_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _postprocess_game(invalid=True, reward=0.0, texts=["bad", "bad"])
    samples += _postprocess_game(invalid=False, reward=1.0, texts=["<move>e2e4</move>"], rollout_id=1)
    args = make_args(custom_rollout_log_function_path="chess_training.log_rollout_metrics")
    logged = Mock()
    monkeypatch.setattr("miles.ray.rollout.metrics.tracking.log", logged)
    extra = {"existing_metric": 7.0}
    log_rollout_data(12, args, samples, extra, 1.0)
    logged.assert_called_once()
    metrics = logged.call_args.args[1]
    assert metrics["rollout/chess/invalid_move_termination_count"] == 1
    assert metrics["rollout/chess/trajectory_count"] == 2
    assert metrics["rollout/chess/invalid_move_termination_rate"] == 0.5
    assert metrics["rollout/num_training_samples"] == 3
    assert metrics["rollout/episode_raw_reward"] == 0.5
    assert metrics["existing_metric"] == 7.0
    assert "rollout/step" in metrics
    assert logged.call_args.kwargs["step_key"] == "rollout/step"
    assert args.custom_rollout_log_function_path == "chess_training.log_rollout_metrics"
    assert extra == {"existing_metric": 7.0}


def test_no_failure_emits_zero_and_accepts_no_extra_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _postprocess_game(invalid=False, reward=1.0, texts=["<move>e2e4</move>"])
    logged = Mock()
    monkeypatch.setattr("miles.ray.rollout.metrics.tracking.log", logged)
    args = make_args(custom_rollout_log_function_path="chess_training.log_rollout_metrics")
    log_rollout_data(0, args, samples, None, 1.0)
    assert logged.call_args.args[1]["rollout/chess/invalid_move_termination_rate"] == 0.0


@pytest.mark.parametrize("flag", [None, "false", 0, 1, [], {}])
def test_missing_or_non_boolean_termination_metadata_fails_explicitly(flag: Any) -> None:
    sample = make_sample(metadata={"chess_result": {"invalid_move_termination": flag}})
    with pytest.raises(ValueError, match="boolean invalid_move_termination"):
        chess_training.log_rollout_metrics(0, make_args(), [sample], {}, 1.0)


def test_conflicting_sibling_flags_are_rejected() -> None:
    samples = _postprocess_game(invalid=False, reward=1.0, texts=["first", "second"])
    samples[1].metadata["chess_result"] = {"invalid_move_termination": True}
    with pytest.raises(ValueError, match="Sibling samples disagree"):
        chess_training.log_rollout_metrics(0, make_args(), samples, {}, 1.0)


@pytest.mark.parametrize("penalty", [-1.0, float("nan"), float("inf"), True, "0.5"])
def test_bad_penalties_are_rejected(penalty: Any) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        chess_training.postprocess_samples(
            [],
            {"agent": {"chess_result": {"invalid_move_termination": False}, "repetition_reward_penalty": penalty}},
        )
