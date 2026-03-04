from tests.fast.utils.ft.conftest import (
    EMPTY_RANK_PLACEMENT,
    inject_training_job_status,
    make_fake_metric_store,
    make_fake_mini_wandb,
)

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models import ActionType

# Numeric values match controller._JOB_STATUS_TO_NUMERIC
_RUNNING = 1
_STOPPED = 0
_FAILED = -1
_PENDING = 2


class TestTrainingCrashDetector:
    def test_job_running(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_RUNNING)
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_job_failed_crash(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_FAILED)
        wandb = make_fake_mini_wandb(steps={1: {"loss": 2.5}})
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, wandb, EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"

    def test_job_failed_nan_loss(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_FAILED)
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("nan")}})
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, wandb, EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_job_failed_inf_loss(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_FAILED)
        wandb = make_fake_mini_wandb(steps={1: {"loss": float("inf")}})
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, wandb, EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "nan_loss"

    def test_job_pending(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_PENDING)
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_job_stopped(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_STOPPED)
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, make_fake_mini_wandb(), EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.NONE

    def test_job_failed_no_loss_data(self) -> None:
        store = make_fake_metric_store()
        inject_training_job_status(store, status_value=_FAILED)
        wandb = make_fake_mini_wandb()
        detector = TrainingCrashDetector()

        decision = detector.evaluate(store, wandb, EMPTY_RANK_PLACEMENT)

        assert decision.action == ActionType.ENTER_RECOVERY
        assert decision.trigger == "crash"
