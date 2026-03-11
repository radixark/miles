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
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
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
