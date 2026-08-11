from __future__ import annotations

import argparse
import logging
import signal
import sys
from dataclasses import dataclass
from time import sleep
from types import FrameType

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.process_utils import launch_bound_subprocess
from miles.utils.workers.serving.utils import split_worker_argv

logger = logging.getLogger(__name__)

_KEEP_ALIVE_POLL_SECONDS = 60
_UNINSTALL_JOB_RETRY_SLEEPS = (5, 15, 45, 90)


def main(argv: list[str] | None = None) -> int:
    configure_logger_raw("orchestrator_wrapper")
    own_argv, command = split_worker_argv(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Run the orchestration script and publish its exit code")
    parser.add_argument("--state-file", required=True, help="Path of the orchestrator exit file to write")
    parser.add_argument(
        "--uninstall-manifest",
        default=None,
        help="Path of a rendered job manifest that uninstalls this run's release once its verdict is in",
    )
    args = parser.parse_args(own_argv)
    assert command, "Pass the orchestration script after a -- separator"

    runner = _Runner(state_file=args.state_file, uninstall_manifest=args.uninstall_manifest)
    exit_code = runner.run(command)

    logger.info(f"exit_code={exit_code}, staying alive so the logs remain readable")
    _keep_alive()
    return exit_code


def _keep_alive() -> None:
    while True:
        sleep(_KEEP_ALIVE_POLL_SECONDS)


@dataclass(frozen=True)
class _Runner:
    state_file: str
    uninstall_manifest: str | None

    def run(self, command: list[str]) -> int:
        self._install_signal_verdict()

        if (decided := self._handle_previous_state()) is not None:
            return decided

        self._publish(OrchestratorStatus.STARTED, exit_code=None)
        logger.info(f"Running {command}")

        try:
            process = launch_bound_subprocess(command, envs={})
            exit_code = _exit_code_of_returncode(process.wait())
        except SystemExit:
            logger.info("The signal handler already published this run's verdict; leaving it alone")
            raise
        except BaseException as exception:
            logger.error(f"Failed to run the orchestration script: {exception!r}", exc_info=True)
            self._conclude(exit_code=1)
            raise

        return self._conclude(exit_code=exit_code)

    def _install_signal_verdict(self) -> None:
        def handle(signal_number: int, frame: FrameType | None) -> None:
            if self._is_verdict_published():
                logger.info(f"Received signal {signal_number} after the run reported its exit code; exiting")
            else:
                logger.info(f"Received signal {signal_number}; publishing a verdict before exiting")
                self._publish(OrchestratorStatus.EXITED, exit_code=128 + signal_number)
            raise SystemExit(128 + signal_number)

        for signal_number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signal_number, handle)

    def _handle_previous_state(self) -> int | None:
        """The exit code this pod is already bound to, or None when it is the first to run this launch."""
        previous = OrchestratorState.read(self.state_file)
        if previous is None:
            return None

        if previous.is_terminal and previous.exit_code is not None:
            logger.info(f"This run already reported exit_code={previous.exit_code}; not running it again")
            self._create_uninstall_job()
            return previous.exit_code

        if not previous.is_terminal:
            logger.error(
                f"This pod restarted while the orchestration script was still running "
                f"(status {previous.status.value}); reporting the run as failed instead of training a second time"
            )
            return self._conclude(exit_code=1)

        return None

    def _is_verdict_published(self) -> bool:
        published = OrchestratorState.read(self.state_file)
        return published is not None and published.is_terminal and published.exit_code is not None

    def _conclude(self, *, exit_code: int) -> int:
        self._publish(OrchestratorStatus.EXITED, exit_code=exit_code)
        self._create_uninstall_job()
        return exit_code

    def _create_uninstall_job(self) -> None:
        if self.uninstall_manifest is None:
            return

        for attempt, sleep_seconds in enumerate(_UNINSTALL_JOB_RETRY_SLEEPS, start=1):
            if self._create_uninstall_job_once(attempt=attempt):
                return
            sleep(sleep_seconds)

        logger.error(
            f"Gave up creating the uninstall job of {self.uninstall_manifest} after "
            f"{len(_UNINSTALL_JOB_RETRY_SLEEPS)} attempts, so this release stays installed until it is "
            f"uninstalled by hand"
        )

    def _create_uninstall_job_once(self, *, attempt: int) -> bool:
        try:
            created = Kubectl.create_if_absent(self.uninstall_manifest)
        except Exception:
            logger.error(
                f"Attempt {attempt} at creating the uninstall job of {self.uninstall_manifest} failed", exc_info=True
            )
            return False

        if created:
            logger.info(f"This run's release uninstalls itself through the job created from {self.uninstall_manifest}")
        else:
            logger.info("This run's uninstall job already exists, so an earlier attempt created it")
        return True

    def _publish(self, status: OrchestratorStatus, *, exit_code: int | None) -> None:
        OrchestratorState(status=status, exit_code=exit_code).write(self.state_file)


def _exit_code_of_returncode(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


if __name__ == "__main__":
    raise SystemExit(main())
