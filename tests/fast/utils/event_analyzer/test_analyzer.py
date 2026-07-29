"""Tests for event_analyzer/analyzer.py."""

from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.event_analyzer.analyzer import run_analysis, run_analysis_from_args
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    InferenceEngineWeightChecksumEvent,
    TrainEngineLocalWeightChecksumEvent,
    TrainEngineLocalWeightChecksumState,
)
from miles.utils.audit_utils.process_identity import MainProcessIdentity, TrainProcessIdentity


def _log_checksum_event(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    param_hashes: dict[str, str] | None = None,
) -> None:
    event_logger.log(
        TrainEngineLocalWeightChecksumEvent,
        dict(
            rollout_id=rollout_id,
            state=TrainEngineLocalWeightChecksumState(
                param_hashes=param_hashes or {},
                buffer_hashes={},
                optimizer_hashes=[],
            ),
        ),
    )


def _make_source(*, cell_index: int = 0, rank: int = 0) -> TrainProcessIdentity:
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank)


class TestRunAnalysis:
    def test_empty_directory_returns_no_issues(self, tmp_path: Path) -> None:
        assert run_analysis(event_dir=tmp_path) == []

    def test_delegates_to_rules_and_returns_issues(self, tmp_path: Path) -> None:
        # Cross-replica rule compares same rank across DIFFERENT cells, so use cell_index 0/1.
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(cell_index=0, rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(cell_index=1, rank=0))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "zzz"})
        logger_b.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1


def _log_inference_engine_checksum_event(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    engine_checksums: list[dict[str, str]],
) -> None:
    event_logger.log(
        InferenceEngineWeightChecksumEvent,
        dict(rollout_id=rollout_id, engine_checksums=engine_checksums),
    )


class TestInferenceEngineChecksumRuleWiredIn:
    def test_engine_inconsistency_reported(self, tmp_path: Path) -> None:
        """run_analysis surfaces engine-to-engine checksum mismatches via the registered rule."""
        event_logger = EventLogger(log_dir=tmp_path, file_name="e.jsonl", source=MainProcessIdentity())
        _log_inference_engine_checksum_event(
            event_logger, rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}]
        )
        event_logger.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1

    def test_consistent_engines_no_issue(self, tmp_path: Path) -> None:
        """Identical engine checksums produce no issue."""
        event_logger = EventLogger(log_dir=tmp_path, file_name="e.jsonl", source=MainProcessIdentity())
        _log_inference_engine_checksum_event(
            event_logger, rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]
        )
        event_logger.close()

        assert run_analysis(event_dir=tmp_path) == []


class TestRunAnalysisFromArgs:
    def test_skips_when_disabled(self) -> None:
        args = Namespace(enable_event_analyzer=False, save_debug_event_data="/tmp/whatever")
        run_analysis_from_args(args)

    def test_skips_when_no_event_dir(self) -> None:
        args = Namespace(enable_event_analyzer=True)
        run_analysis_from_args(args)

    def test_raises_on_mismatch(self, tmp_path: Path) -> None:
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(cell_index=0, rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(cell_index=1, rank=0))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "zzz"})
        logger_b.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        with pytest.raises(ValueError, match="issues"):
            run_analysis_from_args(args)

    def test_passes_when_all_match(self, tmp_path: Path) -> None:
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(rank=1))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_b.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        run_analysis_from_args(args)
