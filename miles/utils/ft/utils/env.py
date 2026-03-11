"""Centralized access to MILES_FT_* environment variables.

All env-var reads in miles.utils.ft go through the functions here so that
the mapping from env-var name to Python value is defined in one place.
Functions (not module-level constants) are used because some env vars may be
set after import time.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_ft_id() -> str:
    return os.environ.get("MILES_FT_ID", "")


def get_training_run_id() -> str:
    return os.environ.get("MILES_FT_TRAINING_RUN_ID", "")


def get_exception_inject_path() -> Path | None:
    raw = os.environ.get("MILES_FT_EXCEPTION_INJECT_PATH", "")
    return Path(raw) if raw else None
