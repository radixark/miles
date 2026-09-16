from __future__ import annotations

from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

IMPORTED_MODULES_ENV_VAR = "MILES_E2E_IMPORTED_MODULES"
ENV_VAR_FN_FAILURE_MESSAGE = "env var hook refuses to run"


def compute_env_vars(argv: list[str]) -> dict[str, str]:
    return {IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules()}


def raise_env_var_error(argv: list[str]) -> dict[str, str]:
    raise RuntimeError(ENV_VAR_FN_FAILURE_MESSAGE)
