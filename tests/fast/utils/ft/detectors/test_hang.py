from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.helpers import (
    EMPTY_RANK_PLACEMENT,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.metric_names import (
    TRAINING_ITERATION,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.detectors.hang import HangDetector
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.models import ActionType, MetricSample
from miles.utils.ft.platform.protocols import JobStatus


def _inject_iteration(
    store: MiniPrometheus,
    value: float,
    timestamp: datetime | None = None,
    rank: str = "0",
) -> None:
    store.ingest_samples(
        target_id=f"rank-{rank}",
        samples=[MetricSample(name=TRAINING_ITERATION, labels={"rank": rank}, value=value)],
        timestamp=timestamp,
    )


def _inject_phase(
    store: MiniPrometheus,
    phase: float,
    rank: str = "0",
) -> None:
    store.ingest_samples(
        target_id=f"rank-{rank}",
        samples=[MetricSample(name=TRAINING_PHASE, labels={"rank": rank}, value=phase)],
    )


class TestHangDetector:
    def test_iteration_progressing(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=110.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(training_timeout_minutes=10)
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_iteration_stalled(self) -> None:
        store = make_fake_metric_store()
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(training_timeout_minutes=10)
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "training" in decision.reason

    def test_checkpoint_saving_stalled(self) -> None:
        """Checkpoint saving: iteration stalled within lookback window → hang detected."""
        store = make_fake_metric_store()
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_checkpoint_saving_not_hung(self) -> None:
        """Checkpoint saving: iteration changed within 30min window → not hung."""
        store = make_fake_metric_store()
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=20))
        _inject_iteration(store, value=101.0, timestamp=now - timedelta(minutes=15))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_checkpoint_saving_hang(self) -> None:
        """Checkpoint saving: iteration stalled for entire 30min window."""
        store = make_fake_metric_store()
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=25))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=10))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_job_not_running(self) -> None:
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.FAILED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_no_iteration_data(self) -> None:
        store = make_fake_metric_store()
        detector = HangDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
