from __future__ import annotations

import json
from pathlib import Path

import pytest

from miles.utils.debug_utils.run_megatron.logprob_comparator import (
    _CompareResult,
    _PositionLogprob,
    _compute_comparison,
    _load_and_merge,
    compare_logprobs,
)


def _make_rank_json(
    entries_by_batch: list[list[dict]],
) -> str:
    return json.dumps({"entries": entries_by_batch})


def _write_rank_file(
    directory: Path,
    rank: int,
    entries_by_batch: list[list[dict]],
) -> Path:
    path = directory / f"rank_{rank}.json"
    path.write_text(_make_rank_json(entries_by_batch))
    return path


def _entry(
    global_position: int,
    token_id: int,
    logprob: float,
    is_valid: bool = True,
) -> dict:
    return {
        "global_position": global_position,
        "token_id": token_id,
        "logprob": logprob,
        "is_valid": is_valid,
    }


class TestLoadAndMerge:
    def test_single_rank_single_batch(self, tmp_path: Path) -> None:
        _write_rank_file(tmp_path, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0), _entry(1, 101, -2.0)],
        ])
        result = _load_and_merge(tmp_path)
        assert len(result) == 2
        assert (0, 0) in result
        assert (0, 1) in result
        assert result[(0, 0)].logprob == -1.0

    def test_multi_batch(self, tmp_path: Path) -> None:
        _write_rank_file(tmp_path, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0)],
            [_entry(0, 200, -3.0)],
        ])
        result = _load_and_merge(tmp_path)
        assert (0, 0) in result
        assert (1, 0) in result
        assert result[(0, 0)].token_id == 100
        assert result[(1, 0)].token_id == 200

    def test_is_valid_false_skipped(self, tmp_path: Path) -> None:
        _write_rank_file(tmp_path, rank=0, entries_by_batch=[
            [
                _entry(0, 100, -1.0, is_valid=True),
                _entry(1, 101, -2.0, is_valid=False),
            ],
        ])
        result = _load_and_merge(tmp_path)
        assert len(result) == 1
        assert (0, 0) in result
        assert (0, 1) not in result

    def test_tp_deduplication(self, tmp_path: Path) -> None:
        entries = [_entry(0, 100, -1.0)]
        _write_rank_file(tmp_path, rank=0, entries_by_batch=[entries])
        _write_rank_file(tmp_path, rank=1, entries_by_batch=[entries])
        result = _load_and_merge(tmp_path)
        assert len(result) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = _load_and_merge(tmp_path)
        assert result == {}

    def test_no_matching_files(self, tmp_path: Path) -> None:
        (tmp_path / "other.json").write_text("{}")
        result = _load_and_merge(tmp_path)
        assert result == {}


class TestComputeComparison:
    def _make_entries(
        self, data: list[tuple[int, int, float]]
    ) -> dict[tuple[int, int], _PositionLogprob]:
        return {
            (0, pos): _PositionLogprob(global_position=pos, token_id=tid, logprob=lp)
            for pos, tid, lp in data
        }

    def test_identical_passes(self) -> None:
        entries = self._make_entries([(0, 100, -1.0), (1, 101, -2.0)])
        result = _compute_comparison(
            baseline_entries=entries,
            target_entries=entries,
            threshold=1e-3,
        )
        assert result.passed is True
        assert result.max_abs_diff == 0.0
        assert result.num_positions == 2

    def test_within_threshold_passes(self) -> None:
        baseline = self._make_entries([(0, 100, -1.0)])
        target = self._make_entries([(0, 100, -1.0005)])
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=1e-3,
        )
        assert result.passed is True
        assert result.max_abs_diff == pytest.approx(0.0005)

    def test_exceeds_threshold_fails(self) -> None:
        baseline = self._make_entries([(0, 100, -1.0)])
        target = self._make_entries([(0, 100, -1.1)])
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=0.01,
        )
        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(0.1)

    def test_no_common_keys_passes(self) -> None:
        baseline = self._make_entries([(0, 100, -1.0)])
        target: dict[tuple[int, int], _PositionLogprob] = {
            (1, 0): _PositionLogprob(global_position=0, token_id=200, logprob=-2.0)
        }
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=1e-3,
        )
        assert result.passed is True
        assert result.num_positions == 0

    def test_statistics(self) -> None:
        baseline = self._make_entries([
            (0, 100, -1.0),
            (1, 101, -2.0),
            (2, 102, -3.0),
        ])
        target = self._make_entries([
            (0, 100, -1.01),
            (1, 101, -2.02),
            (2, 102, -3.03),
        ])
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=1.0,
        )
        assert result.passed is True
        assert result.num_positions == 3
        assert result.max_abs_diff == pytest.approx(0.03)
        assert result.mean_abs_diff == pytest.approx(0.02)
        assert result.median_abs_diff == pytest.approx(0.02)

    def test_worst_position_tracked(self) -> None:
        baseline = self._make_entries([
            (0, 100, -1.0),
            (1, 101, -2.0),
        ])
        target = self._make_entries([
            (0, 100, -1.0),
            (1, 101, -2.5),
        ])
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=1.0,
        )
        assert result.max_diff_position == 1
        assert result.max_diff_token_id == 101
        assert result.max_diff_baseline_logprob == pytest.approx(-2.0)
        assert result.max_diff_target_logprob == pytest.approx(-2.5)

    def test_threshold_boundary_equal_passes(self) -> None:
        baseline = self._make_entries([(0, 100, -1.0)])
        target = self._make_entries([(0, 100, -1.001)])
        result = _compute_comparison(
            baseline_entries=baseline,
            target_entries=target,
            threshold=0.001,
        )
        assert result.passed is True


class TestCompareLogprobs:
    def test_both_have_data_passes(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        entries = [_entry(0, 100, -1.0), _entry(1, 101, -2.0)]
        _write_rank_file(baseline_dir, rank=0, entries_by_batch=[entries])
        _write_rank_file(target_dir, rank=0, entries_by_batch=[entries])

        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            threshold=1e-3,
        ) is True

    def test_large_diff_fails(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _write_rank_file(baseline_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0)],
        ])
        _write_rank_file(target_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -2.0)],
        ])

        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            threshold=0.01,
        ) is False

    def test_empty_baseline_returns_true(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()
        _write_rank_file(target_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0)],
        ])

        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
        ) is True

    def test_empty_target_returns_true(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()
        _write_rank_file(baseline_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0)],
        ])

        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
        ) is True

    def test_custom_threshold(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _write_rank_file(baseline_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.0)],
        ])
        _write_rank_file(target_dir, rank=0, entries_by_batch=[
            [_entry(0, 100, -1.05)],
        ])

        # 0.05 > 0.01 → fail
        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            threshold=0.01,
        ) is False

        # 0.05 <= 0.1 → pass
        assert compare_logprobs(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            threshold=0.1,
        ) is True
