"""CI metric-history collection backend.

Captures a fixed set of training/rollout metrics from the live process (driver or
Ray actor) into a per-process, append-only NDJSON record. The record is a pure
process-to-harness handoff: it carries only ``{metric_key: [(step, value), ...]}``
series, never identity (no test path), never reads wandb, and never writes to any
cloud. Reduction and gating happen in a later step that consumes these records.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from typing import Any

from .base import TrackingBackend

logger = logging.getLogger(__name__)

# Metric keys captured for the history gate, plus the step key carried alongside
# each. The training keys are logged from the Ray training actor with step_key
# "train/step"; rollout/raw_reward is logged with step_key "rollout/step". Keys
# are the actor (role="actor") form with no role prefix.
TARGET_METRIC_KEYS: tuple[str, ...] = (
    "train/grad_norm",
    "train/ppo_kl",
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_kl",
    "rollout/raw_reward",
)

# Env var naming the directory the harness assigns for this run's records.
RECORD_DIR_ENV = "MILES_CI_GATE_RECORD_DIR"


class CiHistoryBackend(TrackingBackend):
    """Accumulate target metrics in-process and persist the raw series to disk.

    One instance lives per process that runs ``init_tracking`` (the driver and
    each main-rank actor). Each instance owns a distinct NDJSON file keyed by a
    fresh process-local id, so concurrent Ray processes never clobber each other.

    The target metrics are logged from the Ray training actor, whose ``finish()``
    is never called (``finish_tracking()`` runs only on the driver). So every
    ``log()`` persists a fresh snapshot of the full accumulated series; the
    file is the latest snapshot regardless of whether ``finish()`` ever fires.
    """

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[int | None, float]]] = {}
        self._lock = threading.Lock()
        self._record_dir: str | None = None
        self._record_path: str | None = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        record_dir = os.environ.get(RECORD_DIR_ENV)
        if not record_dir:
            # No harness-assigned directory: nothing to collect into. Leaving
            # _record_dir None makes log()/finish() no-ops.
            logger.info("%s not set; CI history collection disabled.", RECORD_DIR_ENV)
            return
        os.makedirs(record_dir, exist_ok=True)
        self._record_dir = record_dir
        process_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._record_path = os.path.join(record_dir, f"{process_id}.ndjson")

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        if self._record_dir is None:
            return
        with self._lock:
            captured = False
            for key in TARGET_METRIC_KEYS:
                if key not in metrics:
                    continue
                value = metrics[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                self._series.setdefault(key, []).append((step, float(value)))
                captured = True
            if captured:
                self._write_snapshot_locked()

    def finish(self) -> None:
        if self._record_dir is None:
            return
        with self._lock:
            self._write_snapshot_locked()

    def _write_snapshot_locked(self) -> None:
        # Rewrite the whole per-process file with the current series. Writing to a
        # temp file and renaming makes each snapshot atomic, so a concurrent reader
        # (the harness merge) never sees a half-written record.
        assert self._record_path is not None
        tmp_path = f"{self._record_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for key, points in self._series.items():
                line = {
                    "metric": key,
                    "series": [[step, value] for step, value in points],
                }
                f.write(json.dumps(line) + "\n")
        os.replace(tmp_path, self._record_path)
