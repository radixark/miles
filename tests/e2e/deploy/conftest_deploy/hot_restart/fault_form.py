import logging
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    compute_hot_restart_workloads,
    read_restart_stamp_of_workload,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_hot_restart_config, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, read_run_progress
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

logger = logging.getLogger(__name__)

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0
BASELINE_STAMP_READ_ATTEMPTS: int = 10


class HotRestartFaultForm(BaseFaultForm):
    def __init__(
        self,
        *,
        launch: Callable[[ExecuteTrainConfig], None],
        config: ExecuteTrainConfig,
        checkpoint_dir: Path,
        events_dir: Path,
        poll_interval_seconds: float = TAKE_OVER_POLL_INTERVAL_SECONDS,
        timeout_seconds: float = TAKE_OVER_TIMEOUT_SECONDS,
        baseline_read_attempts: int = BASELINE_STAMP_READ_ATTEMPTS,
    ) -> None:
        self._launch = launch
        self._config = config
        self._release = compute_release_of_config(config)
        self._namespace = config.namespace
        self._checkpoint_dir = checkpoint_dir
        self._events_dir = events_dir
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._baseline_read_attempts = baseline_read_attempts
        self._stamped_workloads = compute_hot_restart_workloads(self._release)
        self._threads: list[threading.Thread] = []
        self._failures: list[tuple[int, BaseException]] = []
        self._records: list[HotRestartRecord] = []

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def records(self) -> tuple[HotRestartRecord, ...]:
        return tuple(self._records)

    def join_relaunches(self, *, timeout_seconds: float = RELAUNCH_JOIN_TIMEOUT_SECONDS) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout_seconds)

    def assert_take_overs_installed_cleanly(self) -> None:
        assert not self._failures, "a hot restart of this run did not install cleanly:\n" + "\n".join(
            f"  - take-over {at}: {failure!r}" for at, failure in self._failures
        )
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        assert not alive, (
            f"{alive} are still installing a hot restart, so this run may still be replaced under the dumps that "
            f"are about to be read"
        )

    @property
    def harms_cell(self) -> bool:
        return False

    def inject(self, cell: dict, rng: random.Random) -> None:
        progress = read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        stamps_before = self._read_stamps_to_replace()

        logger.info(f"Hot restarting {self._release} of a run that stands at {progress}")
        index = len(self._threads)
        self._relaunch_on_thread()
        self._wait_until_take_over_reached_run(index=index, stamps_before=stamps_before)
        self._records.append(
            HotRestartRecord(
                index=index,
                saved_iteration_at_trigger=progress.last_saved_iteration,
                frozen_rollout_id=(
                    -1 if progress.last_finished_rollout_id is None else progress.last_finished_rollout_id
                ),
            )
        )

    def _relaunch_on_thread(self) -> None:
        index = len(self._threads)
        thread = threading.Thread(target=self._relaunch, args=(index,), daemon=True, name=f"take-over-{index}")
        self._threads.append(thread)
        thread.start()

    def _relaunch(self, index: int) -> None:
        try:
            self._launch(compute_hot_restart_config(self._config, installed_release=self._release))
        except BaseException as e:
            logger.warning(f"The hot restart of {self._release} failed to install", exc_info=True)
            self._failures.append((index, e))

    def _wait_until_take_over_reached_run(self, *, index: int, stamps_before: dict[str, str | None]) -> None:
        deadline = time.monotonic() + self._timeout_seconds
        relaunch = self._threads[index]

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval_seconds)
            if restamped_replaced_workloads(
                before=stamps_before, after=self._read_restart_stamps_or_none(), workloads=self._stamped_workloads
            ):
                logger.info(f"The take-over of {self._release} landed; it restamped {sorted(self._stamped_workloads)}")
                return

            assert not self._failures_of(index), (
                f"the hot restart of {self._release} was refused rather than installed, so the run keeps training "
                f"under the script this injection meant to replace: {self._failures_of(index)}"
            )
            assert relaunch.is_alive(), (
                f"the relaunch of {self._release} returned without restamping {sorted(self._stamped_workloads)}, so "
                f"nothing took its orchestration script over"
            )

        raise AssertionError(
            f"the relaunch of {self._release} was installed {self._timeout_seconds}s ago and "
            f"{sorted(self._stamped_workloads)} still carry the stamps they carried before it ({stamps_before}), "
            f"so a take-over cannot be told from a relaunch that hung"
        )

    def _read_stamps_to_replace(self) -> dict[str, str | None]:
        for _ in range(self._baseline_read_attempts):
            if (stamps := self._read_restart_stamps_or_none()) is not None:
                return stamps
            time.sleep(self._poll_interval_seconds)

        raise AssertionError(
            f"the workloads of {self._release} could not be read in {self._baseline_read_attempts} attempt(s), and "
            f"a take-over is told from a relaunch that hung by the stamps it replaces, so this draw has nothing to "
            f"compare against"
        )

    def _failures_of(self, index: int) -> list[BaseException]:
        return [failure for at, failure in self._failures if at == index]

    def _read_restart_stamps_or_none(self) -> dict[str, str | None] | None:
        try:
            return read_restart_stamp_of_workload(release=self._release, namespace=self._namespace)
        except Exception:
            logger.warning(f"Failed to read the restart stamps of {self._release}", exc_info=True)
            return None


def restamped_replaced_workloads(
    *, before: dict[str, str | None], after: dict[str, str | None] | None, workloads: frozenset[str]
) -> bool:
    if after is None:
        return False
    return all((stamp := after.get(one)) is not None and stamp != before.get(one) for one in workloads)
