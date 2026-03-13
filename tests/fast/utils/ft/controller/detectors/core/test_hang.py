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
from miles.utils.ft.controller.types import ActionType


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
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_no_iteration_data(self) -> None:
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),

            job_status=JobStatus.RUNNING,
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE


# ---------------------------------------------------------------------------
# P2 item 16: _get_current_phase() edge cases
# ---------------------------------------------------------------------------


class TestGetCurrentPhaseEdgeCases:
    def test_no_metric_data_returns_noop(self) -> None:
        """_get_current_phase() returns PHASE_TRAINING (default) when no data → short-circuit."""
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.NONE

    def test_heartbeat_exists_but_phase_missing(self) -> None:
        """Heartbeat metric exists but phase metric missing → defaults to training phase."""
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=5))
        inject_heartbeat(store, value=100.0, timestamp=now - timedelta(minutes=1))

        detector = HangDetector(config=HangDetectorConfig(training_timeout_minutes=10))
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),
            job_status=JobStatus.RUNNING,
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.ENTER_RECOVERY
        assert "training" in decision.reason

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
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY


class TestHangDetectorAmbiguousPhaseSeries:
    def test_ambiguous_phase_defaults_to_training(self) -> None:
        """When multiple TRAINING_PHASE series exist for rank=0 (e.g. stale
        + current exporters coexist), _get_current_phase used to pick
        row(0) nondeterministically. Now uses query_single_latest which
        raises AmbiguousSeriesError; the detector catches it and defaults
        to PHASE_TRAINING instead of using an arbitrary value."""
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
        )

        decision = detector.evaluate(ctx)
        # Defaults to training phase timeout, heartbeat stalled → recovery
        assert decision.action == ActionType.ENTER_RECOVERY
        assert "training" in decision.reason


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
        )

        decision = detector.evaluate(ctx)
        assert decision.action == ActionType.ENTER_RECOVERY


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

    def test_no_run_id_in_context_queries_all_runs_for_backwards_compat(self) -> None:
        """When active_run_id is None (e.g. older controller code path),
        the detector must still work by querying without ft_run_id filter."""
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

        assert decision.action == ActionType.ENTER_RECOVERY
