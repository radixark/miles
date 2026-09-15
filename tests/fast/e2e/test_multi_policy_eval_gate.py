from pathlib import Path

import pytest

import tests.e2e.conftest_multi_policy as conftest_multi_policy
from tests.e2e.conftest_multi_policy import EvalScoreBounds


def _patch_scores(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> None:
    monkeypatch.setattr(conftest_multi_policy, "_read_eval_score_series", lambda *_args, **_kwargs: scores)


def _assert_learned(bounds: EvalScoreBounds) -> None:
    conftest_multi_policy.assert_policy_eval_scores_learned(
        Path("unused"), bounds={"solver": bounds}, dataset_name="gsm8k"
    )


def test_peak_not_last_point_passes(monkeypatch):
    """A series that dips after its best point is judged on the peak, not the final value."""
    _patch_scores(monkeypatch, [0.45, 0.55, 0.60, 0.52])
    _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.58, min_growth=0.1))


def test_starting_solved_fails(monkeypatch):
    """A first point above initial_max fails because it cannot demonstrate learning."""
    _patch_scores(monkeypatch, [0.9, 0.95])
    with pytest.raises(AssertionError, match="starts already solved"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))


def test_peak_below_threshold_fails(monkeypatch):
    """A peak below peak_min fails the accuracy gate."""
    _patch_scores(monkeypatch, [0.4, 0.45])
    with pytest.raises(AssertionError, match="peak eval score"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))


def test_growth_below_threshold_fails(monkeypatch):
    """A peak-minus-first gap below min_growth fails the growth gate."""
    _patch_scores(monkeypatch, [0.55, 0.60])
    with pytest.raises(AssertionError, match="eval growth"):
        _assert_learned(EvalScoreBounds(initial_max=0.6, peak_min=0.58, min_growth=0.2))


def test_none_growth_skips_growth_gate(monkeypatch):
    """min_growth=None requires no growth at all."""
    _patch_scores(monkeypatch, [0.6, 0.6])
    _assert_learned(EvalScoreBounds(initial_max=0.65, peak_min=0.58, min_growth=None))


def test_no_points_fails(monkeypatch):
    """An empty eval series fails loudly instead of passing vacuously."""
    _patch_scores(monkeypatch, [])
    with pytest.raises(AssertionError, match="no eval/gsm8k/solver points"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))
