from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.ft.utils import (
    inject_heartbeat,
    inject_training_phase,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.utils.metric_names import AGENT_HEARTBEAT, TRAINING_PHASE
from miles.utils.ft.controller.types import ActionType, TriggerType


class TestHangDetector:
    def test_iteration_progressing(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=110.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_iteration_stalled(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "training" in decision.reason

    def test_checkpoint_saving_stalled(self) -> None:
        """Checkpoint saving: iteration stalled within lookback window → hang detected."""
        store = make_fake_metric_store()
        inject_training_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(
            config=HangDetectorConfig(
                training_timeout_minutes=10,
                checkpoint_saving_timeout_minutes=30,
            )
        )
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_checkpoint_saving_not_hung(self) -> None:
        """Checkpoint saving: iteration changed within 30min window → not hung."""
        store = make_fake_metric_store()
        inject_training_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=20))
        inject_heartbeat(store, value=101.0, timestamp=now - timedelta(minutes=15))
        detector = HangDetector(
            config=HangDetectorConfig(
                training_timeout_minutes=10,
                checkpoint_saving_timeout_minutes=30,
            )
        )
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_checkpoint_saving_hang(self) -> None:
        """Checkpoint saving: iteration stalled for entire 30min window."""
        store = make_fake_metric_store()
        inject_training_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=25))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=10))
        detector = HangDetector(
            config=HangDetectorConfig(
                training_timeout_minutes=10,
                checkpoint_saving_timeout_minutes=30,
            )
        )
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_job_not_running(self) -> None:
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.FAILED,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_no_iteration_data_skips_hang_check(self) -> None:
        """Empty metric store means phase is unknown. The detector skips hang
        check rather than defaulting to training timeout, which would misfire
        during checkpoint saving."""
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason


# ---------------------------------------------------------------------------
# P2 item 16: _get_current_phase() edge cases
# ---------------------------------------------------------------------------


class TestGetCurrentPhaseEdgeCases:
    def test_no_metric_data_skips_hang_check(self) -> None:
        """_get_current_phase() returns None when no data. The detector skips
        the hang check entirely rather than defaulting to training timeout,
        which would misfire during checkpoint saving."""
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason

    def test_heartbeat_exists_but_phase_missing_skips_hang_check(self) -> None:
        """Heartbeat metric exists but phase metric is missing. Previously
        defaulted to training timeout, which could misidentify a normal
        checkpoint save (>10min) as a hang. Now skips hang check."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason

    def test_unknown_phase_value_returns_noop(self) -> None:
        """Phase metric with unknown value (not training/checkpoint) → skip hang check."""
        store = make_fake_metric_store()
        inject_training_phase(store, phase=99.0)
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE
        assert "unknown" in decision.reason.lower()


class TestHangDetectorSingleSampleFalsePositive:
    def test_single_heartbeat_sample_does_not_trigger_hang(self) -> None:
        """With only 1 heartbeat sample, changes() returns 0 but that does not
        mean the heartbeat is stalled — there is simply insufficient data.
        Previously this would trigger ENTER_RECOVERY (false positive)."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_two_samples_same_value_does_trigger_hang(self) -> None:
        """With 2+ samples showing the same value, hang should be detected."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY


class TestHangDetectorAmbiguousPhaseSeries:
    def test_ambiguous_phase_skips_hang_check(self) -> None:
        """When multiple TRAINING_PHASE series exist for rank=0 (e.g. stale
        + current exporters coexist), _get_current_phase returns None.
        Previously defaulted to PHASE_TRAINING, which would apply the
        shorter training timeout and could misfire during checkpoint saving.
        Now skips hang check entirely."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Two phase series for rank=0 from different nodes
        inject_training_phase(store, phase=1.0, rank="0", timestamp=now - timedelta(seconds=5))
        store.ingest_samples(
            target_id="node-stale",
            samples=[GaugeSample(name=TRAINING_PHASE, labels={"rank": "0"}, value=2.0)],
            timestamp=now - timedelta(seconds=3),
        )

        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason


class TestHangDetectorMultipleRank0Series:
    def test_hang_detected_when_one_rank0_series_stalled(self) -> None:
        """H-7: when multiple rank-0 series exist (e.g. stale + current),
        the detector previously used df['value'][0] which could pick the
        progressing series and miss the stalled one. Now uses min()."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Series A (node-a): progressing
        store.ingest_samples(
            target_id="node-a",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=100.0)],
            timestamp=now - timedelta(minutes=5),
        )
        store.ingest_samples(
            target_id="node-a",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=200.0)],
            timestamp=now - timedelta(minutes=1),
        )

        # Series B (node-b): stalled — zero changes
        store.ingest_samples(
            target_id="node-b",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=50.0)],
            timestamp=now - timedelta(minutes=5),
        )
        store.ingest_samples(
            target_id="node-b",
            samples=[GaugeSample(name=AGENT_HEARTBEAT, labels={"rank": "0"}, value=50.0)],
            timestamp=now - timedelta(minutes=1),
        )

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.ENTER_RECOVERY


# ---------------------------------------------------------------------------
# Telemetry blind: "no data" vs "insufficient data" distinction
# ---------------------------------------------------------------------------


class TestHangDetectorTelemetryBlind:
    """Verify that HangDetector distinguishes "completely no heartbeat data"
    (TELEMETRY_BLIND → NOTIFY_HUMAN) from "data exists but insufficient"
    (no_fault, wait for more samples).

    Previously, both cases returned no_fault with reason "no heartbeat data
    available", treating monitoring blindness as healthy. This masked scenarios
    where the rank-0 agent never started, the metric scrape pipeline broke,
    or the run_id didn't match."""

    def test_empty_store_skips_hang_check(self) -> None:
        """Metric store has zero data after grace period. Phase is unknown so
        hang check is skipped rather than defaulting to training timeout."""
        store = make_fake_metric_store()
        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="run-1",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason

    def test_single_sample_returns_insufficient_not_blind(self) -> None:
        """One heartbeat sample exists (data is arriving) but changes() is 0
        because there's only 1 point. This is "insufficient data", not
        "telemetry blind". Previously both returned no_fault with the same
        reason string — now they are distinguished."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "insufficient" in decision.reason

    def test_two_samples_progressing_returns_healthy(self) -> None:
        """Two heartbeat samples with different values → healthy."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=200.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "progressing" in decision.reason

    def test_two_samples_stalled_returns_hang(self) -> None:
        """Two heartbeat samples with same value → ENTER_RECOVERY with HANG trigger."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.HANG

    def test_run_id_mismatch_returns_telemetry_blind(self) -> None:
        """Heartbeat data exists for run-1 but context has active_run_id=run-2.
        Label filter excludes all data → TELEMETRY_BLIND. Previously returned
        no_fault, hiding the run_id mismatch."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5), ft_run_id="run-1")
        inject_heartbeat(store, value=200.0, timestamp=now - timedelta(minutes=1), ft_run_id="run-1")

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="run-2",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert decision.trigger == TriggerType.TELEMETRY_BLIND


class TestHangDetectorPhaseUnknownSkipsCheck:
    """Verify that when phase metric is missing or ambiguous, the hang
    detector does not default to training timeout. Previously, fallback
    to PHASE_TRAINING (10min) would misfire during checkpoint saving
    (which can legitimately take >10min)."""

    def test_phase_missing_during_checkpoint_save_does_not_trigger_false_hang(self) -> None:
        """The critical scenario: checkpoint save is happening but phase metric
        drops. Previously the detector would fall back to training timeout
        (10min) and flag the normal long save as a hang."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Heartbeat stalled for 15min — normal during checkpoint save,
        # but would exceed training_timeout_minutes=10.
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=15))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        ))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason

    def test_ambiguous_phase_does_not_apply_training_timeout(self) -> None:
        """AmbiguousSeriesError should not default to training timeout."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        inject_training_phase(store, phase=1.0, rank="0", timestamp=now - timedelta(seconds=5))
        store.ingest_samples(
            target_id="node-stale",
            samples=[GaugeSample(name=TRAINING_PHASE, labels={"rank": "0"}, value=2.0)],
            timestamp=now - timedelta(seconds=3),
        )

        # Heartbeat stalled 15min: exceeds training timeout but within ckpt timeout
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=15))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        ))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
        assert "phase unknown" in decision.reason

    def test_known_phase_still_detects_hang(self) -> None:
        """Phase known + heartbeat stalled → still triggers recovery."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_training_phase(store, phase=1.0)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="test-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == TriggerType.HANG


class TestHangDetectorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(training_timeout_minutes=0), "must be >= 1"),
            (dict(training_timeout_minutes=-5), "must be >= 1"),
            (dict(checkpoint_saving_timeout_minutes=0), "must be >= 1"),
            (dict(checkpoint_saving_timeout_minutes=-10), "must be >= 1"),
        ],
    )
    def test_invalid_parameter_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            HangDetectorConfig(**kwargs)


# ---------------------------------------------------------------------------
# Run isolation: ft_run_id label filtering
# ---------------------------------------------------------------------------


class TestHangDetectorRunIsolation:
    """Verify that HangDetector only considers metrics belonging to the
    current run when active_run_id is set on DetectorContext.

    Before ft_run_id filtering was added, stale heartbeat/phase data from
    a previous run could remain in the metric store after a run switch and
    pollute hang-detection decisions for the new run."""

    def test_phase_query_only_sees_current_run(self) -> None:
        """Same rank=0, different ft_run_id: detector must only read the
        current run's phase metric."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Old run: phase = checkpoint_saving (would use 30min timeout)
        inject_training_phase(store, phase=2.0, rank="0", ft_run_id="old-run")

        # Current run: phase = training (uses 10min timeout)
        inject_training_phase(store, phase=1.0, rank="0", ft_run_id="current-run")

        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5), ft_run_id="current-run")
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1), ft_run_id="current-run")

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="current-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert "training" in decision.reason

    def test_heartbeat_query_only_sees_current_run(self) -> None:
        """Old run's stalled heartbeat must not trigger recovery when the
        current run's heartbeat is progressing."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Old run: stalled heartbeat
        inject_heartbeat(store, value=50.0, timestamp=now - timedelta(minutes=5), ft_run_id="old-run")
        inject_heartbeat(store, value=50.0, timestamp=now - timedelta(minutes=1), ft_run_id="old-run")

        # Current run: progressing heartbeat
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5), ft_run_id="current-run")
        inject_heartbeat(store, value=200.0, timestamp=now - timedelta(minutes=1), ft_run_id="current-run")

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="current-run",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_late_scrape_from_old_run_does_not_affect_current_run(self) -> None:
        """After a run switch, a late-arriving scrape from the old run is
        stored with the old ft_run_id. The detector must ignore it."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)

        # Current run: progressing
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5), ft_run_id="run-2")
        inject_heartbeat(store, value=200.0, timestamp=now - timedelta(minutes=1), ft_run_id="run-2")

        # Late scrape from old run arrives *after* the current run's data
        inject_heartbeat(store, value=42.0, timestamp=now - timedelta(seconds=30), ft_run_id="run-1")

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id="run-2",
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_no_run_id_in_context_returns_telemetry_blind(self) -> None:
        """When active_run_id is None, run-scoped metric queries cannot be
        reliable (label filter without ft_run_id would match stale data from
        any run). Previously this fell through and queried all runs for
        backwards compatibility, but that hid run_id mismatch issues.
        Now short-circuits to TELEMETRY_BLIND so humans investigate."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
            active_run_id=None,
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert decision.trigger == TriggerType.TELEMETRY_BLIND
        assert "active_run_id" in decision.reason
