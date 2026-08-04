"""CPU tests for the (sum, count)-aware DP metric reduction."""

from __future__ import annotations

import math

from miles.backends.training_utils.log_utils import reduce_gathered_log_dict


def test_scalar_values_keep_legacy_mean():
    gathered = [{"a": 1.0}, {"a": 3.0}]
    assert reduce_gathered_log_dict(gathered, dp_size=2) == {"a": 2.0}


def test_tuple_values_weighted_by_count():
    # rank0: 3 samples with sum 3.0; rank1: 1 sample with sum 5.0
    gathered = [{"m": (3.0, 3)}, {"m": (5.0, 1)}]
    out = reduce_gathered_log_dict(gathered, dp_size=2)
    assert math.isclose(out["m"], 8.0 / 4)


def test_tuple_equal_counts_match_mean_of_means():
    """With equal per-rank counts (today's invariant) the weighted average must
    equal the legacy mean-of-per-rank-means, so metrics stay comparable."""
    sums = [2.0, 6.0, 4.0, 8.0]
    count = 4
    gathered = [{"m": (s, count)} for s in sums]
    out = reduce_gathered_log_dict(gathered, dp_size=4)
    legacy = sum(s / count for s in sums) / 4
    assert math.isclose(out["m"], legacy)


def test_zero_total_count_maps_to_zero():
    gathered = [{"m": (0.0, 0)}, {"m": (0.0, 0)}]
    assert reduce_gathered_log_dict(gathered, dp_size=2) == {"m": 0.0}


def test_mixed_keys():
    gathered = [
        {"scalar": 1.0, "weighted": (2.0, 2)},
        {"scalar": 3.0, "weighted": (6.0, 2)},
    ]
    out = reduce_gathered_log_dict(gathered, dp_size=2)
    assert out["scalar"] == 2.0
    assert math.isclose(out["weighted"], 2.0)
