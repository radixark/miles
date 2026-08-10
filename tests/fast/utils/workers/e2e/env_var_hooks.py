from __future__ import annotations

from tests.fast.utils.workers.e2e.e2e_worker import spec_of
from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

from miles.utils.workers.worker_spec import ServeWorkerSpec

IMPORTED_MODULES_ENV_VAR = "MILES_E2E_IMPORTED_MODULES"
ENV_VAR_FN_FAILURE_MESSAGE = "env var hook refuses to run"


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [
        spec_of(worker_argv, env_var=lambda context: {IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules()})
    ]


def compute_failing_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [spec_of(worker_argv, env_var=_raise)]


def _raise(context) -> dict[str, str]:
    raise RuntimeError(ENV_VAR_FN_FAILURE_MESSAGE)
