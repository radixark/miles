import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, RunProgress, read_run_progress

from miles.utils.test_utils.ft_test_actions import SLEEP_FOREVER_AT_END_ACTION, read_frozen_rollout_id
from miles.utils.test_utils.polling_worker import PollingWorker


# ================================= constants ==================================


logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 5.0
FREEZE_TIMEOUT_SECONDS: float = 3600.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0
CONSECUTIVE_READ_FAILURE_LIMIT: int = 20
TRACKER_SETTLE_ATTEMPTS: int = 3
TRACKER_SETTLE_INTERVAL_SECONDS: float = 2.0


# ============================== run directories ===============================


CHECKPOINT_DIRNAME: str = "checkpoints"


def compute_checkpoint_dir(dump_dir: str) -> Path:
    return Path(dump_dir) / CHECKPOINT_DIRNAME


# ============================== the freeze plan ===============================


@dataclass(frozen=True)
class ScheduledFreeze:
    frozen_rollout_id: int
    saved_iteration: int | None

    def __post_init__(self) -> None:
        assert self.frozen_rollout_id >= 0, f"a run starts at step 0 and cannot freeze after {self.frozen_rollout_id}"
        assert self.saved_iteration is None or self.saved_iteration < self.frozen_rollout_id, (
            f"a run frozen after step {self.frozen_rollout_id} resumes from iteration {self.saved_iteration}, "
            f"which already covers that step, so nothing would be redone"
        )


def compute_freeze_plan(frozen_rollout_id: int | None) -> list[dict]:
    if frozen_rollout_id is None:
        return []
    return [{"at_rollout": frozen_rollout_id, "action": SLEEP_FOREVER_AT_END_ACTION}]


# ============================== the take-over loop =============================


@dataclass
class HotRestartDriver:
    relaunch: Callable[[int | None], None]
    checkpoint_dir: Path
    events_dir: Path
    # TODO ad hoc hack: revert after the args refactor
    freeze_plan_path: Path
    schedule: tuple[ScheduledFreeze, ...]
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    freeze_timeout_seconds: float = FREEZE_TIMEOUT_SECONDS
    read_failure_limit: int = CONSECUTIVE_READ_FAILURE_LIMIT
    tracker_settle_attempts: int = TRACKER_SETTLE_ATTEMPTS
    tracker_settle_interval_seconds: float = TRACKER_SETTLE_INTERVAL_SECONDS
    records: list[HotRestartRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert (
            self.schedule
        ), "an empty schedule describes a run nothing freezes, and this driver only takes over a frozen run"
        self._failures: list[BaseException] = []
        self._relaunch_threads: list[threading.Thread] = []
        self._worker = PollingWorker(name="hot-restart-driver", run=self._drive)
        self._max_finished_rollout_id: int | None = None

    @property
    def num_restarts(self) -> int:
        return len(self.schedule)

    def start(self) -> None:
        for path in (self.checkpoint_dir, self.events_dir):
            assert not path.exists(), (
                f"{path} exists before this run was even installed: it would resume from a previous run and be compared "
                f"against a baseline that started from nothing"
            )
        self._worker.start()

    def stop_collecting(self) -> None:
        self._worker.stop_and_join(timeout_seconds=RELAUNCH_JOIN_TIMEOUT_SECONDS)
        for thread in self._relaunch_threads:
            thread.join(timeout=RELAUNCH_JOIN_TIMEOUT_SECONDS)

    def assert_nothing_running(self) -> None:
        self._worker.assert_not_running(
            message=(
                f"the hot restart driver was still working {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop, so reading what it collected would race it"
            )
        )
        for thread in self._relaunch_threads:
            assert not thread.is_alive(), (
                f"{thread.name} is still installing a hot restart {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after the run "
                f"ended, so the dumps about to be compared may still be replaced under it"
            )

    def assert_all_restarts_happened(self) -> None:
        assert not self._failures, "the hot restart driver failed:\n" + "\n".join(
            f"  - {one!r}" for one in self._failures
        )
        assert len(self.records) == self.num_restarts, (
            f"the run ended after {len(self.records)} of {self.num_restarts} hot restart(s), so what a take-over "
            f"of a still-training run's trainers costs was never measured"
        )

    def _drive(self, stop_event: threading.Event) -> None:
        try:
            self._take_over_at_scheduled_freezes(stop_event)
        except BaseException as e:
            logger.warning("The hot restart driver stopped before every take-over had landed", exc_info=True)
            self._failures.append(e)

    def _take_over_at_scheduled_freezes(self, stop_event: threading.Event) -> None:
        for index, scheduled in enumerate(self.schedule):
            progress = self._wait_until_run_frozen(stop_event, index=index, scheduled=scheduled)
            if progress is None:
                return

            record = self._compute_record(index=index, scheduled=scheduled, progress=progress)
            self.records.append(record)
            logger.info(f"Hot restart {record.index} is due against a frozen run: {record}")
            self._relaunch_on_thread(index)

    def _wait_until_run_frozen(
        self, stop_event: threading.Event, *, index: int, scheduled: ScheduledFreeze
    ) -> RunProgress | None:
        deadline = time.monotonic() + self.freeze_timeout_seconds
        failures = 0

        while not stop_event.is_set():
            if (progress := self._read_progress()) is None:
                failures += 1
                assert failures < self.read_failure_limit, (
                    f"hot restart {index} failed to read how far the run had come {failures} time(s) in a row, so "
                    f"it cannot tell a run frozen after step {scheduled.frozen_rollout_id} from one still training"
                )
            else:
                failures = 0
                self._assert_no_step_lost_outside_take_over(progress)
                if self._stands_frozen_at(scheduled, progress=progress):
                    return progress

            assert time.monotonic() < deadline, (
                f"hot restart {index} waited {self.freeze_timeout_seconds}s for a run frozen after step "
                f"{scheduled.frozen_rollout_id}, and the run only reached {progress}"
            )
            stop_event.wait(timeout=self.poll_interval_seconds)

        return None

    def _read_progress(self) -> RunProgress | None:
        try:
            return read_run_progress(checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir)
        except Exception:
            logger.warning("Failed to read how far the run being hot restarted has come", exc_info=True)
            return None

    def _stands_frozen_at(self, scheduled: ScheduledFreeze, *, progress: RunProgress) -> bool:
        if (finished := progress.last_finished_rollout_id) is not None:
            assert finished <= scheduled.frozen_rollout_id, (
                f"the run was armed to sleep after step {scheduled.frozen_rollout_id} and finished step {finished}: "
                f"the injection that freezes it never fired, so a take-over would race the run"
            )

        return read_frozen_rollout_id(self.freeze_plan_path) == scheduled.frozen_rollout_id

    def _compute_record(self, *, index: int, scheduled: ScheduledFreeze, progress: RunProgress) -> HotRestartRecord:
        saved = self._read_settled_saved_iteration(on=scheduled.saved_iteration, progress=progress)

        assert saved == scheduled.saved_iteration, (
            f"the run frozen after step {scheduled.frozen_rollout_id} holds iteration {saved}, not the pinned "
            f"{scheduled.saved_iteration}: it saves at another save cadence than it was installed with, so the "
            f"take-over resumes from a checkpoint nobody pinned"
        )
        return HotRestartRecord(
            index=index,
            saved_iteration_at_trigger=scheduled.saved_iteration,
            frozen_rollout_id=scheduled.frozen_rollout_id,
        )

    def _read_settled_saved_iteration(self, *, on: int | None, progress: RunProgress) -> int | None:
        saved = progress.last_saved_iteration
        for _ in range(self.tracker_settle_attempts):
            if saved == on:
                return saved
            time.sleep(self.tracker_settle_interval_seconds)
            if (reread := self._reread_saved_iteration()) is not None:
                saved = reread
        return saved

    def _reread_saved_iteration(self) -> int | None:
        try:
            return read_run_progress(
                checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir
            ).last_saved_iteration
        except Exception:
            logger.warning("Failed to re-read the checkpoint tracker of a run being hot restarted", exc_info=True)
            return None

    def _assert_no_step_lost_outside_take_over(self, progress: RunProgress) -> None:
        finished = progress.last_finished_rollout_id
        if finished is None:
            return
        if self.records and finished <= self.records[-1].frozen_rollout_id:
            return

        assert self._max_finished_rollout_id is None or finished >= self._max_finished_rollout_id, (
            f"the run had finished step {self._max_finished_rollout_id} and now reports {finished}: outside a "
            f"take-over's rollback an event log only grows, so this run lost work nobody asked it to"
        )
        self._max_finished_rollout_id = finished

    def _relaunch_on_thread(self, index: int) -> None:
        thread = threading.Thread(
            target=self._relaunch,
            args=(self._compute_next_frozen_rollout_id(index),),
            daemon=True,
            name=f"hot-restart-relaunch-{index}",
        )
        self._relaunch_threads.append(thread)
        thread.start()

    def _compute_next_frozen_rollout_id(self, index: int) -> int | None:
        if index + 1 >= len(self.schedule):
            return None
        return self.schedule[index + 1].frozen_rollout_id

    def _relaunch(self, frozen_rollout_id: int | None) -> None:
        try:
            self.relaunch(frozen_rollout_id)
        except BaseException as e:
            logger.warning("A hot restart relaunch failed", exc_info=True)
            self._failures.append(e)
