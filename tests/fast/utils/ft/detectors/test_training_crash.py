from tests.fast.utils.ft.helpers import (
    EMPTY_RANK_PLACEMENT,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models import ActionType
from miles.utils.ft.platform.protocols import JobStatus


class TestTrainingCrashDetector:
    def test_job_running(self) -> None:
        store = make_fake_metric_store()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_job_failed_crash(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=wandb, rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.FAILED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"

    def test_job_failed_nan_loss(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=wandb, rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.FAILED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_job_failed_inf_loss(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("inf")}})
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=wandb, rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.FAILED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_job_pending(self) -> None:
        store = make_fake_metric_store()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.PENDING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_job_stopped(self) -> None:
        store = make_fake_metric_store()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.STOPPED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE

    def test_job_failed_no_loss_data(self) -> None:
        store = make_fake_metric_store()
        wandb = make_fake_mini_wandb()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=wandb, rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.FAILED)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"

    def test_empty_metric_store(self) -> None:
        store = make_fake_metric_store()
        detector = TrainingCrashDetector()
        ctx = make_detector_context(metric_store=store, mini_wandb=make_fake_mini_wandb(), rank_placement=EMPTY_RANK_PLACEMENT, job_status=JobStatus.RUNNING)

        decision = detector.evaluate(ctx)

        assert decision.action == ActionType.NONE
