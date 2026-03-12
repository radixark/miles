import pytest
from tests.fast.utils.ft.utils import (
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ActionType


class TestTrainingCrashDetector:
    @pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.STOPPED])
    def test_non_failed_status_is_noop(self, status: JobStatus) -> None:
        store = make_fake_metric_store()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store,
            mini_wandb=make_fake_mini_wandb(),

            job_status=status,
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_job_failed_crash(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.FAILED
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"

    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_job_failed_with_non_finite_loss(self, bad_loss: float) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": bad_loss}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.FAILED
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_job_failed_no_loss_data(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.FAILED
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"


# ---------------------------------------------------------------------------
# P2 item 15: _determine_trigger() branching isolation
# ---------------------------------------------------------------------------


class TestDetermineTrigger:
    """Isolate _determine_trigger() branching (P2 item 15)."""

    def test_job_failed_with_nan_loss_returns_nan_loss_trigger(self) -> None:
        """Job FAILED + NaN loss → TriggerType.NAN_LOSS (prioritized over CRASH)."""
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.FAILED
        )

        decision = detector.evaluate(ctx)

        assert decision.trigger == "nan_loss"

    def test_job_failed_without_nan_loss_returns_crash_trigger(self) -> None:
        """Job FAILED + no NaN → TriggerType.CRASH."""
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.FAILED
        )

        decision = detector.evaluate(ctx)

        assert decision.trigger == "crash"

    def test_job_running_with_nan_loss_returns_noop(self) -> None:
        """Job RUNNING + NaN loss → NONE (crash detector only fires on FAILED)."""
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(
            metric_store=store, mini_wandb=wandb, job_status=JobStatus.RUNNING
        )

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
