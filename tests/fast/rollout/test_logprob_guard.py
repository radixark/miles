from typing import Any

import pytest

from miles.rollout.logprob_guard import guard_rollout_log_probs
from miles.utils.types import Sample


def _sample(log_probs: list[Any] | None) -> Sample:
    response_length = len(log_probs) if log_probs is not None else 2
    return Sample(
        tokens=list(range(response_length)),
        response_length=response_length,
        rollout_log_probs=log_probs,
    )


def test_guard_accepts_finite_and_missing_log_probs():
    samples = [_sample([-0.1, -0.2]), _sample(None)]

    guard_rollout_log_probs(samples, [[1, 1], [1, 1]])


def test_guard_ignores_non_finite_masked_tokens():
    sample = _sample([float("nan"), float("inf"), -0.1])

    guard_rollout_log_probs([sample], [[0, 0, 1]])


def test_guard_reports_non_finite_active_tokens():
    samples = [
        _sample([-0.1, float("nan")]),
        _sample([float("inf"), float("-inf")]),
        _sample([float("nan"), -0.2]),
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"3 bad tokens \(active_nan=1, active_inf=2, null=0, invalid_type=0\) "
            r"across 6 total tokens \(5 active\).*row indices: \[0, 1\]"
        ),
    ):
        guard_rollout_log_probs(samples, [[1, 1], [1, 1], [0, 1]])


def test_guard_reports_wire_null_and_invalid_type():
    samples = [_sample([None, -0.1]), _sample(["invalid", -0.2])]

    with pytest.raises(
        ValueError,
        match=(
            r"2 bad tokens \(active_nan=0, active_inf=0, null=1, invalid_type=1\) "
            r"across 4 total tokens \(4 active\).*row indices: \[0, 1\]"
        ),
    ):
        guard_rollout_log_probs(samples, [[1, 1], [1, 1]])


def test_guard_rejects_structurally_invalid_masked_tokens():
    samples = [_sample([None, -0.1]), _sample(["invalid", -0.2])]

    with pytest.raises(ValueError, match=r"null=1, invalid_type=1.*4 total tokens \(2 active\)"):
        guard_rollout_log_probs(samples, [[0, 1], [0, 1]])


def test_guard_rejects_length_mismatch():
    with pytest.raises(ValueError, match=r"rollout_log_probs length \(1\) != loss_mask length \(2\) at row 0"):
        guard_rollout_log_probs([_sample([-0.1])], [[1, 1]])
