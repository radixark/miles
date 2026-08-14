from __future__ import annotations

import argparse
import os
from pathlib import Path


WORKER_FACTORY_ERROR = "e2e worker factory refuses to build a worker"


class E2eWorker:
    def __init__(self, argv: list[str], state_dir: Path) -> None:
        self._argv = argv
        self._state_dir = state_dir

    async def report_pid(self) -> int:
        return os.getpid()

    async def report_argv(self) -> list[str]:
        return self._argv

    async def demo_async(self, value: dict) -> dict:
        return value


def make_worker(argv: list[str]) -> E2eWorker:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    args, _ = parser.parse_known_args(argv)

    return E2eWorker(argv, Path(args.state_dir))


def make_raising_worker(argv: list[str]) -> E2eWorker:
    raise RuntimeError(WORKER_FACTORY_ERROR)
