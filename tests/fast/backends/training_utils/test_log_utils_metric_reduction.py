"""Tests for the per-key metric reduction in log_utils.gather_log_data.

Rank-local minima and maxima must be reduced as extrema across ranks.
Averaging them (the old behavior) systematically under-reports the global
maximum and over-reports the global minimum.
"""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import log_utils


def test_min_max_keys_reduce_to_global_extrema():
    gathered = [
        {
            "raw_response_length/response_length_mean": 120.0,
            "raw_response_length/response_length_max": 400.0,
            "raw_response_length/response_length_min": 8.0,
        },
        {
            "raw_response_length/response_length_mean": 200.0,
            "raw_response_length/response_length_max": 900.0,
            "raw_response_length/response_length_min": 32.0,
        },
    ]
    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={
            "raw_response_length/response_length_max": "max",
            "raw_response_length/response_length_min": "min",
        },
    )

    assert reduced["raw_response_length/response_length_mean"] == pytest.approx(160.0)
    # The old mean reduction would report max=650.0 and min=20.0.
    assert reduced["raw_response_length/response_length_max"] == 900.0
    assert reduced["raw_response_length/response_length_min"] == 8.0


def test_keys_default_to_mean_without_reduction_map():
    gathered = [
        {"custom_metric_max": 2.0},
        {"custom_metric_max": 4.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(gathered, dp_size=2)

    assert reduced == {"custom_metric_max": pytest.approx(3.0)}


def test_explicit_reduction_selects_extrema():
    gathered = [
        {"multi_turn_metric/round_number_max": 2.0},
        {"multi_turn_metric/round_number_max": 4.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={"multi_turn_metric/round_number_max": "max"},
    )

    assert reduced == {"multi_turn_metric/round_number_max": 4.0}


def test_sum_count_tuples_ignore_reduction_map():
    gathered = [
        {"loss": (6.0, 2.0), "score_max": 1.0},
        {"loss": (2.0, 2.0), "score_max": 5.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={"score_max": "max"},
    )

    assert reduced == {"loss": pytest.approx(2.0), "score_max": 5.0}


def test_mismatched_keys_across_ranks_raise():
    gathered = [
        {"multi_turn_metric/round_number_max": 3.0},
        {"multi_turn_metric/round_number_max": 3.0, "extra": 1.0},
    ]

    with pytest.raises(ValueError, match="Metric keys differ across ranks"):
        log_utils.reduce_gathered_log_dict(gathered, dp_size=2, reduction_by_key={})


def test_unknown_reduction_name_raises():
    gathered = [{"multi_turn_metric/round_number_max": 3.0}]

    with pytest.raises(ValueError, match="Unsupported metric reduction"):
        log_utils.reduce_gathered_log_dict(
            gathered,
            dp_size=1,
            reduction_by_key={"multi_turn_metric/round_number_max": "median"},
        )


def test_empty_gather_returns_empty_dict():
    assert log_utils.reduce_gathered_log_dict([], dp_size=2, reduction_by_key={}) == {}


def test_log_multi_turn_data_passes_explicit_extrema_reductions(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(tp=SimpleNamespace(rank=0), is_pp_last_stage=True)
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)

    def capture(metric_name, args, rollout_id, log_dict, reduction_by_key=None):
        captured.update(
            metric_name=metric_name,
            rollout_id=rollout_id,
            log_dict=log_dict,
            reduction_by_key=reduction_by_key,
        )

    monkeypatch.setattr(log_utils, "gather_log_data", capture)

    log_utils.log_multi_turn_data(
        rollout_id=7,
        args=SimpleNamespace(rollout_max_response_len=8),
        rollout_data={
            "loss_masks": [torch.tensor([1, 1, 0]), torch.tensor([1, 1])],
            "round_number": [1, 3],
        },
    )

    assert captured["metric_name"] == "multi_turn"
    assert captured["rollout_id"] == 7
    assert captured["reduction_by_key"] == log_utils._MULTI_TURN_REDUCTION_BY_KEY
