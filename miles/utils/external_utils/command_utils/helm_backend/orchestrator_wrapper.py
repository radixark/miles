from __future__ import annotations

import argparse
import signal
import sys
import time

from miles.utils.external_utils.command_utils.helm_backend.run_state import (
    STATUS_EXITED,
    STATUS_STARTED,
    write_orchestrator_state,
)
from miles.utils.workers.process_utils import launch_bound_subprocess
from miles.utils.workers.serving.utils import split_worker_argv

_KEEP_ALIVE_POLL_SECONDS = 60


def main(argv: list[str] | None = None) -> int:
    own_argv, command = split_worker_argv(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Run the orchestration script and publish its exit code")
    parser.add_argument("--exit-file", required=True, help="Path of the orchestrator exit file to write")
    parser.add_argument(
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stay running after the script exits, so its logs remain readable",
    )
    args = parser.parse_args(own_argv)
    assert command, "pass the orchestration script after a -- separator"

    exit_code = _run(command=command, exit_file=args.exit_file)

    if args.keep_alive:
        _log(f"exit_code={exit_code}, staying alive so the logs remain readable")
        while True:
            time.sleep(_KEEP_ALIVE_POLL_SECONDS)

    return exit_code


def _run(command: list[str], exit_file: str) -> int:
    write_orchestrator_state(exit_file, STATUS_STARTED)
    _install_signal_verdict(exit_file)
    _log(f"running {command}")

    try:
        process = launch_bound_subprocess(command, envs={})
        exit_code = process.wait()
    except SystemExit:
        raise
    except BaseException as exception:
        _log(f"failed to run the orchestration script: {exception!r}")
        write_orchestrator_state(exit_file, STATUS_EXITED, exit_code=1)
        raise

    write_orchestrator_state(exit_file, STATUS_EXITED, exit_code=exit_code)
    return exit_code


def _install_signal_verdict(exit_file: str) -> None:
    def handle(signal_number: int, frame: object) -> None:
        _log(f"received signal {signal_number}; publishing a verdict before exiting")
        write_orchestrator_state(exit_file, STATUS_EXITED, exit_code=128 + signal_number)
        raise SystemExit(128 + signal_number)

    for signal_number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signal_number, handle)


def _log(message: str) -> None:
    print(f"[orchestrator] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
