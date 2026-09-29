"""Tests for event_analyzer/analyzer.py."""

import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_analyzer import analyzer as analyzer_module
from miles.utils.audit_utils.event_analyzer.analyzer import (
    _partition_by_model_id,
    run_analysis,
    run_analysis_from_args,
    run_sample_ownership_analysis,
)
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    InferenceEngineWeightChecksumEvent,
    OutputConsumption,
    SampleLineagePayload,
    TrainEngineLocalWeightChecksumEvent,
    TrainEngineLocalWeightChecksumState,
    TrainerModelCompanionInfoEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


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


class TestSeveralModelIds:
    def test_trainer_controller_events_are_partitioned_by_model_id(self) -> None:
        """Controller events stay with their model while model-neutral events accompany every partition."""
        solver_event = SimpleNamespace(
            source=TrainerControllerProcessIdentity(trainer_id="solver-actor", model_id="solver")
        )
        verifier_event = SimpleNamespace(
            source=TrainerControllerProcessIdentity(trainer_id="verifier-actor", model_id="verifier")
        )
        neutral_event = SimpleNamespace(source=SimpleProcessIdentity(component="main"))

        partitions = _partition_by_model_id([verifier_event, neutral_event, solver_event])

        assert partitions == [[neutral_event, solver_event], [verifier_event, neutral_event]]

    def test_two_model_ids_are_not_compared_against_each_other(self, tmp_path: Path) -> None:
        """Two model ids mean two different models, so their weights differ by design, not by fault."""
        for model_id, param_hash in (("solver", "aaa"), ("verifier", "zzz")):
            for cell_index in (0, 1):
                event_logger = EventLogger(
                    log_dir=tmp_path,
                    file_name=f"{model_id}-{cell_index}.jsonl",
                    source=TrainProcessIdentity(
                        component="actor", model_id=model_id, cell_index=cell_index, rank_within_cell=0
                    ),
                )
                _log_checksum_event(event_logger, rollout_id=0, param_hashes={"pp0.w": param_hash})
                event_logger.close()

        assert run_analysis(event_dir=tmp_path) == []

    def test_two_cells_of_one_model_id_are_still_compared(self, tmp_path: Path) -> None:
        """Partitioning by model id must not disable the check inside one model id, which is what it exists for."""
        for cell_index, param_hash in ((0, "aaa"), (1, "zzz")):
            event_logger = EventLogger(
                log_dir=tmp_path,
                file_name=f"solver-{cell_index}.jsonl",
                source=TrainProcessIdentity(
                    component="actor", model_id="solver", cell_index=cell_index, rank_within_cell=0
                ),
            )
            _log_checksum_event(event_logger, rollout_id=0, param_hashes={"pp0.w": param_hash})
            event_logger.close()

        other = EventLogger(
            log_dir=tmp_path,
            file_name="verifier.jsonl",
            source=TrainProcessIdentity(component="actor", model_id="verifier", cell_index=0, rank_within_cell=0),
        )
        _log_checksum_event(other, rollout_id=0, param_hashes={"pp0.w": "bbb"})
        other.close()

        assert len(run_analysis(event_dir=tmp_path)) == 1


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
        event_logger = EventLogger(
            log_dir=tmp_path, file_name="e.jsonl", source=SimpleProcessIdentity(component="main")
        )
        _log_inference_engine_checksum_event(
            event_logger, rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}]
        )
        event_logger.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1

    def test_consistent_engines_no_issue(self, tmp_path: Path) -> None:
        """Identical engine checksums produce no issue."""
        event_logger = EventLogger(
            log_dir=tmp_path, file_name="e.jsonl", source=SimpleProcessIdentity(component="main")
        )
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

    def test_logs_analysis_duration(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Enabled analysis reports how long its event-log scan takes."""
        caplog.set_level(logging.INFO)

        run_analysis_from_args(Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path)))

        assert f"Event analysis of {tmp_path} took " in caplog.text

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


class TestRunSampleOwnershipAnalysis:
    @staticmethod
    def _args(**overrides: Any) -> Namespace:
        return Namespace(
            **{
                "enable_sample_ownership_checker": True,
                "sample_ownership_grace_steps": 0,
                "ci_test": True,
                **overrides,
            }
        )

    @staticmethod
    def _log_one_completed_step(log_dir: Path, *, count: int) -> None:
        event_logger = EventLogger(log_dir=log_dir, source=SimpleProcessIdentity(component="main"))
        event_logger.log(
            TrainerModelCompanionInfoEvent,
            dict(
                rollout_id=1,
                attempt=0,
                cell_index=0,
                skipped_nonfinite_sample_counts=[],
                sample_counts=[
                    OutputConsumption(
                        sample=SampleLineagePayload(source_sample_index=10, output_index=0, output_count=1),
                        count=count,
                    )
                ],
            ),
            print_log=False,
        )
        event_logger.log(
            TrainGroupStepEndEvent,
            dict(rollout_id=1, attempt=0, role="actor", cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            print_log=False,
        )
        event_logger.close()

    def test_unissued_duplicate_consumption_is_rejected_before_the_first_completed_window(
        self, tmp_path: Path
    ) -> None:
        """Missing issuance history and an incomplete window cannot suppress duplicate consumption."""
        self._log_one_completed_step(tmp_path, count=2)

        with pytest.raises(ValueError, match="count 2"):
            run_sample_ownership_analysis(args=self._args(), event_dir=tmp_path)

    def test_a_violation_outside_ci_is_logged_instead_of_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A training run reports the violation and keeps going; only CI turns it into a failure."""
        self._log_one_completed_step(tmp_path, count=2)

        with caplog.at_level(logging.ERROR):
            run_sample_ownership_analysis(args=self._args(ci_test=False), event_dir=tmp_path)

        assert any(record.levelname == "ERROR" and record.exc_info for record in caplog.records)

    def test_a_disabled_check_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disabled checking does not even open the event log."""
        monkeypatch.setattr(analyzer_module, "read_events", lambda *args, **kwargs: pytest.fail("disabled check ran"))

        run_sample_ownership_analysis(args=self._args(enable_sample_ownership_checker=False))

    def test_the_events_come_from_the_process_event_logger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an explicit directory the check reads what this process has been writing."""
        self._log_one_completed_step(tmp_path, count=1)
        observed: list[int] = []

        def check(events: list[Any], *, grace_steps: int) -> list[Any]:
            observed.append(len(events))
            return []

        monkeypatch.setattr(analyzer_module, "get_event_logger", lambda: SimpleNamespace(log_dir=tmp_path))
        monkeypatch.setattr(analyzer_module.sample_ownership_check, "check", check)

        run_sample_ownership_analysis(args=self._args())

        assert observed == [2]

    def test_analysis_passes_the_step_grace_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The analyzer hands the rule its caller's step grace unchanged."""
        observed: list[int] = []

        def check(_events: list[Any], *, grace_steps: int) -> list[Any]:
            observed.append(grace_steps)
            return []

        monkeypatch.setattr(analyzer_module.sample_ownership_check, "check", check)

        run_sample_ownership_analysis(args=self._args(sample_ownership_grace_steps=3), event_dir=tmp_path)

        assert observed == [3]
