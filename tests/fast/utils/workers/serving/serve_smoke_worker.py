import os

from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

IMPORTED_MODULES_ENV_VAR = "MILES_SERVE_SMOKE_IMPORTED_MODULES"


class SmokeWorker:
    def __init__(self, argv: list[str]):
        self._argv = argv

    def demo_sync(self, a: int, b: int) -> int:
        return a + b

    def report_argv(self) -> list[str]:
        return self._argv

    def report_env(self, name: str) -> str | None:
        return os.environ.get(name)


def make_worker(argv: list[str]) -> SmokeWorker:
    return SmokeWorker(argv)


def compute_env_vars(argv: list[str]) -> dict[str, str]:
    return {
        "MILES_SERVE_SMOKE_ENV": ",".join(argv),
        IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules(),
    }
