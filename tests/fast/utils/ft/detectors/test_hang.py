from datetime import datetime, timedelta, timezone

from tests.fast.utils.ft.conftest import (
    EMPTY_RANK_PLACEMENT,
    inject_training_job_status,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors._metric_names import (
    TRAINING_ITERATION,
    TRAINING_PHASE,
)
from miles.utils.ft.controller.detectors.hang import HangDetector
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus
from miles.utils.ft.models import ActionType, MetricSample

_RUNNING = 1
_FAILED = -1


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
        inject_training_job_status(store, status_value=_RUNNING)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=110.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(training_timeout_minutes=10)

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_iteration_stalled(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(training_timeout_minutes=10)

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "training" in decision.reason

    def test_checkpoint_saving_stalled(self) -> None:
        """Checkpoint saving: iteration stalled within lookback window → hang detected."""
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=5))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=1))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_checkpoint_saving_not_hung(self) -> None:
        """Checkpoint saving: iteration changed within 30min window → not hung."""
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=20))
        _inject_iteration(store, value=101.0, timestamp=now - timedelta(minutes=15))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_checkpoint_saving_hang(self) -> None:
        """Checkpoint saving: iteration stalled for entire 30min window."""
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        _inject_phase(store, phase=2.0)
        now = datetime.now(timezone.utc)
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=25))
        _inject_iteration(store, value=100.0, timestamp=now - timedelta(minutes=10))
        detector = HangDetector(
            training_timeout_minutes=10,
            checkpoint_saving_timeout_minutes=30,
        )

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "hang"
        assert "checkpoint_saving" in decision.reason

    def test_job_not_running(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_FAILED)
        detector = HangDetector()

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_no_iteration_data(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        detector = HangDetector()

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE
